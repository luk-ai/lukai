package lukai

import (
	"context"
	"io"
	"log"
	"runtime"
	"time"

	"github.com/cenkalti/backoff"
	"github.com/luk-ai/lukai/metrics"
	"github.com/luk-ai/lukai/net"
	"github.com/luk-ai/lukai/protobuf/aggregatorpb"
	"github.com/luk-ai/lukai/protobuf/clientpb"
	"github.com/luk-ai/lukai/tf"
	"github.com/luk-ai/lukai/units"
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

	mt.modelCache.Purge()

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

		resp, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		weights := net.ReadModelWeights(func() (*aggregatorpb.ModelWeightChunk, error) {
			resp, err := stream.Recv()
			if err != nil {
				return nil, err
			}
			return resp.GetWeights(), nil
		})
		work := resp.GetWork()
		if work == nil {
			return errors.New("expected work")
		}
		if err := mt.processWork(ctx, c, work, weights); err != nil {
			return errors.Wrapf(err, "failure while processing work: %+v", work.Id)
		}
		// Tensorflow doesn't free memory until the finalizer runs, and GC doesn't
		// run very often since most memory allocations are in C.
		runtime.GC()
	}

	return nil
}

func (mt *ModelType) processWork(
	ctx context.Context,
	c aggregatorpb.EdgeClient,
	work *aggregatorpb.Work,
	weightsReader io.ReadCloser,
) error {
	defer weightsReader.Close()

	log.Printf("Training %+v", work.Id)
	start := time.Now()
	var model *tf.Model
	modelI, ok := mt.modelCache.Get(work.ModelUrl)
	if ok {
		log.Printf("Model from cache")
		model = modelI.(*tf.Model)
	} else {
		var err error
		model, err = tf.GetModel(work.ModelUrl)
		if err != nil {
			return err
		}
		mt.modelCache.Add(work.ModelUrl, model)

		log.Printf("Fetching model took %s", time.Since(start))
	}

	weights, err := tf.LoadWeights(weightsReader)
	if err != nil {
		return errors.Wrapf(err, "loading weights")
	}

	if err := model.SetWeights(weights); err != nil {
		return errors.Wrapf(err, "set weights")
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

	exampleCount := int64(0)

	batchers := batcherCache{}
	defer batchers.Close()

	for i := int64(0); i < work.HyperParams.NumLocalRounds; i++ {
		examples, err := mt.getExampleBatch(batchers, cache, work.HyperParams.BatchSize)
		if err != nil {
			return err
		}
		exampleCount += work.HyperParams.BatchSize

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

		if len(preTrainTargets) > 0 {
			if _, err := model.Session.Run(nil, nil, preTrainTargets); err != nil {
				return errors.Wrapf(
					err, "model.Session.Run(nil, nil, %+v) failed", trainTargets.Pre,
				)
			}
		}

		// Actually train the model on the examples.
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

	if err := model.AddWeights(-1.0, weights); err != nil {
		return err
	}

	var metricVals []float64
	for _, metric := range aggMetrics {
		metricVals = append(metricVals, metric.Val())
	}

	timeTaken := time.Since(start)
	log.Printf("Training took %s", timeTaken)

	stream, err := c.ReportWork(ctx)
	if err != nil {
		return errors.Wrap(err, "grpc ReportWork stream open")
	}
	defer stream.CloseSend()

	if err := stream.Send(&aggregatorpb.ReportWorkRequest{
		Type: &aggregatorpb.ReportWorkRequest_Work{
			&aggregatorpb.Work{
				Id:          work.Id,
				NumExamples: exampleCount,
				NumClients:  1,
				Epoch:       work.Epoch,
				// HyperParams isn't needed here since the server already has that info.
				TimeTaken: timeTaken.Seconds(),
				Started:   work.Started,
				Metrics:   metricVals,
			},
		},
	}); err != nil {
		return errors.Wrap(err, "stream.Send")
	}

	w := net.NewModelWeightsWriter(func(chunk aggregatorpb.ModelWeightChunk) error {
		return stream.Send(&aggregatorpb.ReportWorkRequest{
			Type: &aggregatorpb.ReportWorkRequest_Weights{
				&chunk,
			},
		})
	})

	if err := model.ExportWeights(w); err != nil {
		return errors.Wrapf(err, "model.ExportWeights")
	}
	if err := w.Close(); err != nil {
		return errors.Wrapf(err, "ModelWeightsWriter.Close")
	}

	if _, err := stream.CloseAndRecv(); err != nil {
		return errors.Wrap(err, "stream.CloseAndRecv")
	}

	return nil
}
