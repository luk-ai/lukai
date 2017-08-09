package tf

import tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"

func (model *Model) Weights() (map[tensorflow.Output]*tensorflow.Tensor, error) {
	var variables []tensorflow.Output
	for _, name := range model.TrainableVariables {
		op, n, err := ParseNodeOutput(name)
		if err != nil {
			return nil, err
		}
		variables = append(variables, model.Graph.Operation(op).Output(n))
	}
	results, err := model.Session.Run(
		nil,
		variables,
		nil,
	)
	if err != nil {
		return nil, err
	}
	m := map[tensorflow.Output]*tensorflow.Tensor{}
	for i, output := range variables {
		m[output] = results[i]
	}
	return m, nil
}
