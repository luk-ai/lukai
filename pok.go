package pok

import (
	"sync"
	"time"

	"github.com/pkg/errors"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"

	"github.com/d4l3k/pok/debounce"
	"github.com/d4l3k/pok/protobuf/clientpb"
	"github.com/d4l3k/pok/tf"
)

var ErrNotImplemented = errors.New("not implemented")

const PokAggregatorAddress = "localhost:5000"

var outOfDateModelTimeout = 24 * time.Hour

type ModelType struct {
	Domain, ModelType, DataDir string

	prod struct {
		sync.RWMutex

		model      *tf.Model
		cache      tfOpCache
		lastUpdate time.Time
	}

	training struct {
		sync.Mutex

		running bool
		stop    chan struct{}
	}

	examplesMeta struct {
		sync.RWMutex

		index     clientpb.ExampleIndex
		saveIndex func()
		stop      func()
	}
}

func MakeModelType(domain, modelType, dataDir string) *ModelType {
	mt := ModelType{
		Domain:    domain,
		ModelType: modelType,
		DataDir:   dataDir,
	}
	mt.training.stop = make(chan struct{}, 1)
	mt.examplesMeta.saveIndex, mt.examplesMeta.stop = debounce.Debounce(
		300*time.Second,
		mt.saveExamplesMeta,
	)

	return &mt
}

type tfOpCache struct {
	outputs    map[string]tensorflow.Output
	operations map[string]*tensorflow.Operation
}

func makeTFOpCache() tfOpCache {
	return tfOpCache{
		outputs:    map[string]tensorflow.Output{},
		operations: map[string]*tensorflow.Operation{},
	}
}

func (cache tfOpCache) resolve(
	model *tf.Model,
	ex example,
) (
	map[tensorflow.Output]*tensorflow.Tensor,
	[]tensorflow.Output,
	[]*tensorflow.Operation,
	error,
) {
	var err error

	feeds := map[tensorflow.Output]*tensorflow.Tensor{}
	for name, tensor := range ex.feeds {
		output, ok := cache.outputs[name]
		if !ok {
			output, err = model.Output(name)
			if err != nil {
				return nil, nil, nil, err
			}
			cache.outputs[name] = output
		}

		feeds[output] = tensor
	}

	fetches := []tensorflow.Output{}
	for _, name := range ex.fetches {
		output, ok := cache.outputs[name]
		if !ok {
			output, err = model.Output(name)
			if err != nil {
				return nil, nil, nil, err
			}
			cache.outputs[name] = output
		}

		fetches = append(fetches, output)
	}

	targets := []*tensorflow.Operation{}
	for _, name := range ex.targets {
		op, ok := cache.operations[name]
		if !ok {
			op, err = model.Operation(name)
			if err != nil {
				return nil, nil, nil, err
			}
			cache.operations[name] = op
		}

		targets = append(targets, op)
	}

	return feeds, fetches, targets, nil
}
