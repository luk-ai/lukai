package tf

import tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"

func (model *Model) TrainableVariablesOutputs() ([]tensorflow.Output, error) {
	var outputs []tensorflow.Output
	for _, name := range model.TrainableVariables {
		op, n, err := ParseNodeOutput(name)
		if err != nil {
			return nil, err
		}
		outputs = append(outputs, model.Graph.Operation(model.ApplyPrefix(op)).Output(n))
	}
	return outputs, nil
}

func (model *Model) Weights() (map[tensorflow.Output]*tensorflow.Tensor, error) {
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
	m := map[tensorflow.Output]*tensorflow.Tensor{}
	for i, output := range outputs {
		m[output] = results[i]
	}
	return m, nil
}
