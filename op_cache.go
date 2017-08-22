package pok

import (
	"github.com/d4l3k/pok/tf"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

type tfOpCache struct {
	model      *tf.Model
	outputs    map[string]tensorflow.Output
	operations map[string]*tensorflow.Operation
}

func makeTFOpCache(model *tf.Model) tfOpCache {
	return tfOpCache{
		model:      model,
		outputs:    map[string]tensorflow.Output{},
		operations: map[string]*tensorflow.Operation{},
	}
}

func (cache tfOpCache) resolve(
	ex example,
) (
	map[tensorflow.Output]*tensorflow.Tensor,
	[]tensorflow.Output,
	[]*tensorflow.Operation,
	error,
) {
	feeds, err := cache.resolveFeeds(ex.feeds)
	if err != nil {
		return nil, nil, nil, err
	}

	fetches, err := cache.resolveFetches(ex.fetches)
	if err != nil {
		return nil, nil, nil, err
	}

	targets, err := cache.resolveTargets(ex.targets)
	if err != nil {
		return nil, nil, nil, err
	}

	return feeds, fetches, targets, nil
}

func (cache tfOpCache) resolveFeeds(feedNames map[string]*tensorflow.Tensor) (
	map[tensorflow.Output]*tensorflow.Tensor, error,
) {
	var err error
	feeds := map[tensorflow.Output]*tensorflow.Tensor{}
	for name, tensor := range feedNames {
		output, ok := cache.outputs[name]
		if !ok {
			output, err = cache.model.Output(name)
			if err != nil {
				return nil, err
			}
			cache.outputs[name] = output
		}

		feeds[output] = tensor
	}

	return feeds, nil
}

func (cache tfOpCache) resolveFetches(fetchNames []string) ([]tensorflow.Output, error) {
	var err error
	fetches := []tensorflow.Output{}
	for _, name := range fetchNames {
		output, ok := cache.outputs[name]
		if !ok {
			output, err = cache.model.Output(name)
			if err != nil {
				return nil, err
			}
			cache.outputs[name] = output
		}

		fetches = append(fetches, output)
	}
	return fetches, nil
}

func (cache tfOpCache) resolveTargets(targetNames []string) ([]*tensorflow.Operation, error) {
	var err error
	targets := []*tensorflow.Operation{}
	for _, name := range targetNames {
		op, ok := cache.operations[name]
		if !ok {
			op, err = cache.model.Operation(name)
			if err != nil {
				return nil, err
			}
			cache.operations[name] = op
		}

		targets = append(targets, op)
	}
	return targets, nil
}
