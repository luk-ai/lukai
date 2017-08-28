package tf

import (
	"fmt"

	"github.com/pkg/errors"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// Batcher takes a fixed number of tensors and concatenates them into one larger
// tensor. This is mostly used for creating mini-batches of tensors for SGD.
type Batcher struct {
	shape   []int64
	n       int
	values  []tensorflow.Output
	out     tensorflow.Output
	graph   *tensorflow.Graph
	session *tensorflow.Session
}

// NewTensorBatcher returns a new Batcher with the specified params. Shape should
// have exactly one dimension that is unspecified (-1) and the tensors will be
// concatenated along that axis.
func NewTensorBatcher(n int, dtype tensorflow.DataType, shape []int64) (*Batcher, error) {
	scope := op.NewScope()
	tfShape := tensorflow.MakeShape(shape...)
	var values []tensorflow.Output
	for i := 0; i < n; i++ {
		subScope := scope.SubScope(fmt.Sprintf("input/%d", i))
		val := op.Placeholder(subScope, dtype, op.PlaceholderShape(tfShape))
		values = append(values, val)
	}
	dim := int32(-1)
	for i, size := range shape {
		if dim != -1 && size == -1 {
			return nil, errors.Errorf("shape has more than one undefined axis! %+v", shape)
		}
		if size == -1 {
			dim = int32(i)
		}
	}
	if dim == -1 {
		return nil, errors.Errorf("shape is missing undefined axis! %+v", shape)
	}
	dimOutput := op.Const(scope.SubScope("dim"), dim)
	output := op.Concat(scope, dimOutput, values)
	graph, err := scope.Finalize()
	if err != nil {
		return nil, err
	}
	session, err := tensorflow.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	return &Batcher{
		shape:   shape,
		n:       n,
		values:  values,
		out:     output,
		graph:   graph,
		session: session,
	}, nil
}

// Batch takes in a session and values and returns a single output tensor that
// has all the values concatenated.
func (m *Batcher) Batch(values []*tensorflow.Tensor) (*tensorflow.Tensor, error) {
	if len(values) != m.n {
		return nil, errors.Errorf("expected %d values; got %d", m.n, len(values))
	}
	feeds := map[tensorflow.Output]*tensorflow.Tensor{}
	for i, val := range values {
		feeds[m.values[i]] = val
	}
	out, err := m.session.Run(feeds, []tensorflow.Output{m.out}, nil)
	if err != nil {
		return nil, err
	}
	if len(out) != 1 {
		return nil, errors.Errorf("expected 1 output; got %+v", out)
	}
	return out[0], nil
}

func (m *Batcher) Close() error {
	return m.session.Close()
}
