package pok

import (
	"bytes"
	"context"
	"io"
	"log"
	"time"

	"github.com/cenkalti/backoff"
	"github.com/d4l3k/pok/protobuf/aggregatorpb"
	"github.com/d4l3k/pok/tf"
	"google.golang.org/grpc"
)

// StartTraining starts the training worker.
func (mt *ModelType) StartTraining() error {
	mt.training.Lock()
	defer mt.training.Unlock()

	if mt.training.running {
		return nil
	}

	mt.training.running = true
	go func() {
		backoffOptions := backoff.NewExponentialBackOff()
		backoffOptions.InitialInterval = 1 * time.Second
		backoffOptions.MaxInterval = 1 * time.Minute
		backoffOptions.MaxElapsedTime = 0

		retryCount := 0

		backoff.Retry(func() error {
			select {
			case <-mt.training.stop:
				return nil
			default:
			}

			retryCount += 1

			if err := mt.trainerWorker(); err != nil {
				// TODO(d4l3k): Reconnect on network failure.
				log.Printf("Training error (try #%d). Will retry: %s", retryCount, err)
				return err
			}
			return nil
		}, backoffOptions)

		mt.training.Lock()
		defer mt.training.Unlock()
		mt.training.running = false
	}()

	return nil
}

// StopTraining will asyncronously cause training to stop.
func (mt *ModelType) StopTraining() error {
	mt.training.Lock()
	defer mt.training.Unlock()

	if !mt.training.running {
		return nil
	}

	mt.training.stop <- struct{}{}

	return nil
}

func (mt *ModelType) trainerWorker() error {
	ctx := context.Background()
	// TODO(d4l3k): Secure request
	conn, err := grpc.Dial(PokEdgeAddress, grpc.WithInsecure())
	if err != nil {
		return err
	}
	defer conn.Close()

	c := aggregatorpb.NewEdgeClient(conn)
	stream, err := c.GetWork(ctx, &aggregatorpb.GetWorkRequest{
		Id: aggregatorpb.ModelID{
			Domain:    mt.Domain,
			ModelType: mt.ModelType,
		},
	})
	if err != nil {
		return err
	}

	for {
		select {
		case <-mt.training.stop:
			return nil
		default:
		}

		work, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if err := mt.processWork(ctx, c, work); err != nil {
			return err
		}
	}

	return nil
}

func (mt *ModelType) processWork(
	ctx context.Context, c aggregatorpb.EdgeClient, work *aggregatorpb.Work,
) error {
	start := time.Now()
	model, err := tf.LoadModel(bytes.NewReader(work.Model))
	if err != nil {
		return err
	}
	defer model.Close()

	cache := makeTFOpCache()

	exampleCount := 0

	for i := int64(0); i < work.HyperParams.NumRounds; i++ {
		examples, err := mt.getNExamples(work.HyperParams.BatchSize)
		if err != nil {
			return err
		}
		exampleCount += len(examples)
		// TODO(d4l3k): Implement proper batch training
		for _, example := range examples {
			feeds, fetches, targets, err := cache.resolve(model, example)
			if err != nil {
				return err
			}
			if _, err := model.Session.Run(feeds, fetches, targets); err != nil {
				return err
			}
		}
	}

	var buf bytes.Buffer
	if model.Save(&buf); err != nil {
		return err
	}

	if _, err := c.ReportWork(ctx, &aggregatorpb.ReportWorkRequest{
		Work: aggregatorpb.Work{
			Id:          work.Id,
			NumExamples: int64(exampleCount),
			NumClients:  1,
			Epoch:       work.Epoch,
			Model:       buf.Bytes(),
			// HyperParams isn't needed here since the server already has that info.
			TimeTaken: time.Since(start).Seconds(),
		},
	}); err != nil {
		return err
	}

	return nil
}