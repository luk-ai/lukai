package pok

import (
	"context"
	"io"
	"log"
	"runtime"
	"time"

	"github.com/cenkalti/backoff"
	"github.com/d4l3k/pok/metrics"
	"github.com/d4l3k/pok/protobuf/aggregatorpb"
	"github.com/d4l3k/pok/protobuf/clientpb"
	"github.com/d4l3k/pok/tf"
	"github.com/d4l3k/pok/units"
	"github.com/pkg/errors"
	"google.golang.org/grpc"
)

var MaxMsgSize = 100 * units.MB

// StartTraining starts the training worker.
func (mt *ModelType) StartTraining() error {
	mt.training.Lock()
	defer mt.training.Unlock()

	if mt.training.running {
		return nil
	}

	if mt.TotalExamples() == 0 {
		log.Printf("No training examples available for %s/%s", mt.Domain, mt.ModelType)
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
				log.Printf("Training error (try #%d). Will retry: %+v", retryCount, err)
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
	conn, err := grpc.Dial(
		PokEdgeAddress, grpc.WithInsecure(),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(MaxMsgSize),
			grpc.MaxCallSendMsgSize(MaxMsgSize),
		),
	)
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
			return errors.Wrapf(err, "failure while processing work: %+v", work.Id)
		}
		// Tensorflow doesn't free memory until the finalizer runs, and GC doesn't
		// run very often since most memory allocations are in C.
		runtime.GC()
	}

	return nil
}

func (mt *ModelType) processWork(
	ctx context.Context, c aggregatorpb.EdgeClient, work *aggregatorpb.Work,
) error {
	log.Printf("Training %+v", work.Id)
	start := time.Now()
	model, err := tf.GetModel(work.ModelUrl)
	if err != nil {
		return err
	}
	defer model.Close()

	if err := model.ImportWeights(work.ModelWeights); err != nil {
		return err
	}

	cache := makeTFOpCache(model)

	trainTargets := model.Meta.EventTargets[clientpb.EVENT_TRAIN]
	preTrainTargets, err := cache.resolveTargets(trainTargets.Pre)
	if err != nil {
		return err
	}
	postTrainTargets, err := cache.resolveTargets(trainTargets.Post)
	if err != nil {
		return err
	}

	evalTargets := model.Meta.EventTargets[clientpb.EVENT_EVAL]
	preEvalTargets, err := cache.resolveTargets(evalTargets.Post)
	if err != nil {
		return err
	}
	postEvalTargets, err := cache.resolveTargets(evalTargets.Post)
	if err != nil {
		return err
	}

	var metricFetchNames []string
	var aggMetrics []metrics.Metric
	for _, m := range model.Meta.Metrics {
		metricFetchNames = append(metricFetchNames, m.FetchName)
		aggMetrics = append(aggMetrics, metrics.Get(m.Reduce))
	}
	metricFetches, err := cache.resolveFetches(metricFetchNames)
	if err != nil {
		return err
	}

	exampleCount := 0

	for i := int64(0); i < work.HyperParams.NumLocalRounds; i++ {
		examples, err := mt.getNExamples(work.HyperParams.BatchSize)
		if err != nil {
			return err
		}
		exampleCount += len(examples)

		// Compute training metrics first.
		if len(metricFetches) > 0 {
			if len(preEvalTargets) > 0 {
				if _, err := model.Session.Run(nil, nil, preEvalTargets); err != nil {
					return errors.Wrapf(
						err, "model.Session.Run(nil, nil, %+v) failed", trainTargets.Pre,
					)
				}
			}

			for _, example := range examples {
				feeds, err := cache.resolveFeeds(example.feeds)
				if err != nil {
					return err
				}
				metrics, err := model.Session.Run(feeds, metricFetches, nil)
				if err != nil {
					return errors.Wrapf(
						err, "model.Session.Run(%+v, %+v, %+v) failed",
						example.feeds, example.fetches, example.targets,
					)
				}
				for i, metric := range metrics {
					val, ok := metric.Value().(float64)
					if !ok {
						return errors.Errorf(
							"metric %q datatype not float64! = %+v",
							metricFetchNames, metric.DataType(),
						)
					}
					aggMetrics[i].Add(val)
				}
			}

			if len(postEvalTargets) > 0 {
				if _, err := model.Session.Run(nil, nil, postEvalTargets); err != nil {
					return errors.Wrapf(
						err, "model.Session.Run(nil, nil, %+v) failed", trainTargets.Post,
					)
				}
			}
		}

		// Actually train the model on the examples.

		if len(preTrainTargets) > 0 {
			if _, err := model.Session.Run(nil, nil, preTrainTargets); err != nil {
				return errors.Wrapf(
					err, "model.Session.Run(nil, nil, %+v) failed", trainTargets.Pre,
				)
			}
		}

		// TODO(d4l3k): Implement proper batch training
		for _, example := range examples {
			feeds, fetches, targets, err := cache.resolve(example)
			if err != nil {
				return err
			}
			if _, err := model.Session.Run(feeds, fetches, targets); err != nil {
				return errors.Wrapf(
					err, "model.Session.Run(%+v, %+v, %+v) failed",
					example.feeds, example.fetches, example.targets,
				)
			}
		}

		if len(postTrainTargets) > 0 {
			if _, err := model.Session.Run(nil, nil, postTrainTargets); err != nil {
				return errors.Wrapf(
					err, "model.Session.Run(nil, nil, %+v) failed", trainTargets.Post,
				)
			}
		}
	}

	if err := model.ImportAddWeights(-1.0, work.ModelWeights); err != nil {
		return err
	}

	weights, err := model.ExportWeights()
	if err != nil {
		return err
	}

	var metricVals []float64
	for _, metric := range aggMetrics {
		metricVals = append(metricVals, metric.Val())
	}

	timeTaken := time.Since(start)
	log.Printf("Training took %s", timeTaken)
	if _, err := c.ReportWork(ctx, &aggregatorpb.ReportWorkRequest{
		Work: aggregatorpb.Work{
			Id:           work.Id,
			NumExamples:  int64(exampleCount),
			NumClients:   1,
			Epoch:        work.Epoch,
			ModelWeights: weights,
			// HyperParams isn't needed here since the server already has that info.
			TimeTaken: timeTaken.Seconds(),
			Started:   work.Started,
			Metrics:   metricVals,
		},
	}); err != nil {
		return err
	}

	return nil
}
