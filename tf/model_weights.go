package tf

import (
	"strings"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

func (model *Model) TrainableVariablesOutputs() ([]tensorflow.Output, error) {
	var outputs []tensorflow.Output
	for _, name := range model.Meta.TrainableVariables {
		op, n, err := ParseNodeOutput(name)
		if err != nil {
			return nil, err
		}
		outputs = append(outputs, model.Graph.Operation(model.ApplyPrefix(op)).Output(n))
	}
	return outputs, nil
}

func (model *Model) Weights() ([]*tensorflow.Tensor, error) {
	outputs, err := model.TrainableVariablesOutputs()
	if err != nil {
		return nil, err
	}
	results, err := model.Session.Run(
		nil,
		outputs,
		nil,
	)
	if err != nil {
		return nil, err
	}
	return results, nil
}

const (
	PokAssignOp    = "pok/update/assign"
	PokAssignAddOp = "pok/update/assign_add"
	PokVarPrefix   = "pok/update/var/"
	PokVarScaleOp  = "pok/update/scale"
)

// AddScaledWeights computes:
//   trainable variables += scale * weights
func (model *Model) AddScaledWeights(
	weights []*tensorflow.Tensor, scale *tensorflow.Tensor,
) error {
	feeds := map[tensorflow.Output]*tensorflow.Tensor{
		model.Graph.Operation(PokVarScaleOp).Output(0): scale,
	}
	for i, variable := range model.Meta.TrainableVariables {
		opName := PokVarPrefix + strings.Replace(variable, ":", "/", -1)
		op := model.Graph.Operation(opName)
		feeds[op.Output(0)] = weights[i]
	}

	if _, err := model.Session.Run(
		feeds,
		nil,
		[]*tensorflow.Operation{
			model.Graph.Operation(PokAssignAddOp),
		},
	); err != nil {
		return err
	}

	return nil
}
