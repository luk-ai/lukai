// bindable is a wrapper around lukai that is gomobile/gobind compatible.
//
// gobind is very restricted in what types can be used. Thus, this is a wrapper
// that uses byte arrays for types that can't be used. This isn't intended to be
// used directly, instead should be used via a language specific client wrapper.
//
// See github.com/luk-ai/lukai for details of the functions.
package bindable

import (
	"bytes"
	"context"

	"github.com/luk-ai/lukai"
	"github.com/luk-ai/lukai/tf"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

type ModelType struct {
	*lukai.ModelType
}

func MakeModelType(domain, modelType, dataDir string) (*ModelType, error) {
	mt, err := lukai.MakeModelType(domain, modelType, dataDir)
	if err != nil {
		return nil, err
	}
	return &ModelType{
		ModelType: mt,
	}, nil
}

// Run is a wrapper around ModelType.Log that accepts bytes.
func (mt ModelType) Log(feedsBody, targetsBody []byte) error {
	feeds, err := tf.DecodeTensorMap(bytes.NewReader(feedsBody))
	if err != nil {
		return err
	}
	targets, err := tf.DecodeStringArray(bytes.NewReader(targetsBody))
	if err != nil {
		return err
	}

	return mt.ModelType.Log(feeds, targets)
}

// Run is a wrapper around ModelType.Run that accepts bytes.
func (mt ModelType) Run(feedsBody, fetchesBody, targetsBody []byte) ([]byte, error) {
	feeds, err := tf.DecodeTensorMap(bytes.NewReader(feedsBody))
	if err != nil {
		return nil, err
	}
	fetches, err := tf.DecodeStringArray(bytes.NewReader(fetchesBody))
	if err != nil {
		return nil, err
	}
	targets, err := tf.DecodeStringArray(bytes.NewReader(targetsBody))
	if err != nil {
		return nil, err
	}

	tensors, err := mt.ModelType.Run(context.TODO(), feeds, fetches, targets)
	if err != nil {
		return nil, err
	}
	m := map[string]*tensorflow.Tensor{}
	for i, tensor := range tensors {
		m[fetches[i]] = tensor
	}

	var buf bytes.Buffer
	if err := tf.EncodeTensorMap(&buf, m); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// StartTraining is a wrapper around ModelType that doesn't take a context.
func (mt ModelType) StartTraining() error {
	return mt.ModelType.StartTraining(context.TODO())
}
