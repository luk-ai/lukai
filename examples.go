package pok

import tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"

type example struct {
	feeds   map[string]*tensorflow.Tensor
	fetches []string
	targets []string
}

func (mt *ModelType) Log(feeds map[string]*tensorflow.Tensor, targets []string) error {
	return ErrNotImplemented
}

func (mt *ModelType) getNExamples(n int64) ([]example, error) {
	return nil, ErrNotImplemented
}
