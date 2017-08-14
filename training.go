package pok

import (
	"bytes"
	"context"
	"io"
	"log"

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
		if err := mt.trainerWorker(); err != nil {
			log.Println("Training error:", err)
		}

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

	mt.stopTraining <- struct{}{}

	return nil
}

func (mt *ModelType) trainerWorker() error {
	ctx := context.Background()
	// TODO(d4l3k): Secure request
	conn, err := grpc.Dial(PokAggregatorAddress, grpc.WithInsecure())
	if err != nil {
		return err
	}
	defer conn.Close()

	c := aggregatorpb.NewAggregatorClient(conn)
	stream, err := c.GetWork(ctx, &aggregatorpb.GetWorkRequest{
		Id: []aggregatorpb.ModelID{
			{
				Domain: mt.Domain,
				Name:   mt.ModelType,
			},
		},
	})
	if err != nil {
		return err
	}

	for {
		select {
		case <-mt.stopTraining:
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
	ctx context.Context, c aggregatorpb.AggregatorClient, work *aggregatorpb.Work,
) error {
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
		Work: []aggregatorpb.Work{
			{
				Id:          work.Id,
				NumExamples: int64(exampleCount),
				NumClients:  1,
				Epoch:       work.Epoch,
				Model:       buf.Bytes(),
				// HyperParams isn't needed here since the server already has that info.
			},
		},
	}); err != nil {
		return err
	}

	return nil
}
