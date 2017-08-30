package lukai

import (
	"time"

	context "golang.org/x/net/context"

	"google.golang.org/grpc"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"

	"github.com/luk-ai/lukai/protobuf/aggregatorpb"
	"github.com/luk-ai/lukai/tf"
)

// shouldLoadProdModelRLocked returns whether a new prod model should be loaded.
func (mt *ModelType) shouldLoadProdModelRLocked() bool {
	return mt.prod.model == nil || mt.prod.lastUpdate.Before(time.Now().Add(-outOfDateModelTimeout))
}

// loadProdModelRLocked loads the prod model if it isn't present or is out of date.
func (mt *ModelType) loadProdModelRLocked() error {
	if !mt.shouldLoadProdModelRLocked() {
		return nil
	}
	// Release the read lock, and write lock.
	mt.prod.RUnlock()
	defer mt.prod.RLock()
	mt.prod.Lock()
	defer mt.prod.Unlock()

	// Make sure nothing has changed since we acquired the write lock.
	if !mt.shouldLoadProdModelRLocked() {
		return nil
	}

	conn, err := grpc.Dial(PokEdgeAddress, grpc.WithInsecure())
	if err != nil {
		return err
	}
	defer conn.Close()

	c := aggregatorpb.NewEdgeClient(conn)
	resp, err := c.ProdModel(context.TODO(), &aggregatorpb.ProdModelRequest{
		Id: aggregatorpb.ModelID{
			Domain:    mt.Domain,
			ModelType: mt.ModelType,
		},
	})
	if err != nil {
		return err
	}

	if mt.prod.model != nil {
		if err := mt.prod.model.Close(); err != nil {
			return err
		}
	}

	mt.prod.model, err = tf.GetModel(resp.ModelUrl)
	if err != nil {
		return err
	}
	mt.prod.lastUpdate = time.Now()
	mt.prod.cache = makeTFOpCache(mt.prod.model)

	// TODO(d4l3k): Quantize model weights.
	// TODO(d4l3k): Train with local examples.

	return nil
}

// Run runs the model with the provided tensorflow feeds, fetches and targets.
// The key for feeds, and fetches should be in the form "name:#", and the
// targets in the form "name".
func (mt *ModelType) Run(
	feeds map[string]*tensorflow.Tensor, fetches []string, targets []string,
) ([]*tensorflow.Tensor, error) {
	mt.prod.RLock()
	defer mt.prod.RUnlock()

	feedsResolved, fetchesResolved, targetsResolved, err := mt.prod.cache.resolve(
		example{
			feeds:   feeds,
			fetches: fetches,
			targets: targets,
		},
	)
	if err != nil {
		return nil, err
	}

	return mt.prod.model.Session.Run(feedsResolved, fetchesResolved, targetsResolved)
}
