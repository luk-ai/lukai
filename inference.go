package pok

import (
	"time"

	context "golang.org/x/net/context"

	"google.golang.org/grpc"

	"github.com/pkg/errors"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"

	"github.com/d4l3k/pok/protobuf/aggregatorpb"
	"github.com/d4l3k/pok/tf"
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

	conn, err := grpc.Dial(PokAggregatorAddress, grpc.WithInsecure())
	if err != nil {
		return err
	}
	defer conn.Close()

	c := aggregatorpb.NewAggregatorClient(conn)
	resp, err := c.ProdModel(context.TODO(), &aggregatorpb.ProdModelRequest{
		Ids: []aggregatorpb.ModelID{
			{
				Domain: mt.Domain,
				Name:   mt.ModelType,
			},
		},
	})
	if err != nil {
		return err
	}
	if len(resp.ModelUrls) != 1 {
		return errors.Errorf("server failed to return production model: %+v", resp)
	}

	if mt.prod.model != nil {
		if err := mt.prod.model.Close(); err != nil {
			return err
		}
	}

	mt.prod.model, err = tf.GetModel(resp.ModelUrls[0])
	if err != nil {
		return err
	}
	mt.prod.lastUpdate = time.Now()
	mt.prod.cache = makeTFOpCache()

	return nil
}

func (mt *ModelType) Run(feeds map[string]*tensorflow.Tensor, fetches []string, targets []string) ([]*tensorflow.Tensor, error) {
	mt.prod.RLock()
	defer mt.prod.RUnlock()

	feedsResolved, fetchesResolved, targetsResolved, err := mt.prod.cache.resolve(
		mt.prod.model,
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
