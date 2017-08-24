package tf

import (
	"bytes"
	"encoding/gob"
	"reflect"
	"sort"

	"github.com/pkg/errors"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

func nDimensionalTensor(t reflect.Type, n int) interface{} {
	for i := 0; i < n; i++ {
		t = reflect.SliceOf(t)
	}
	return reflect.New(t).Interface()
}

// EncodeTensor encodes a tensor into a gob.Encoder. See DecodeTensor.
func EncodeTensor(encoder *gob.Encoder, val *tensorflow.Tensor) error {
	if err := encoder.Encode(val.DataType()); err != nil {
		return err
	}
	if err := encoder.Encode(val.Shape()); err != nil {
		return err
	}
	if val.DataType() == tensorflow.String {
		if err := encoder.Encode(val.Value()); err != nil {
			return err
		}
	} else {
		var buf bytes.Buffer
		if _, err := val.WriteContentsTo(&buf); err != nil {
			return err
		}
		if err := encoder.Encode(buf.Bytes()); err != nil {
			return err
		}
	}
	return nil
}

// DecodeTensor decodes a tensor from a gob.Decoder and returns it.
// See EncodeTensor.
func DecodeTensor(decoder *gob.Decoder) (*tensorflow.Tensor, error) {
	var dataType tensorflow.DataType
	if err := decoder.Decode(&dataType); err != nil {
		return nil, errors.Wrap(err, "dataType")
	}
	var shape []int64
	if err := decoder.Decode(&shape); err != nil {
		return nil, errors.Wrap(err, "shape")
	}

	var val *tensorflow.Tensor
	var err error
	if dataType == tensorflow.String {
		valPtr := nDimensionalTensor(reflect.TypeOf(""), len(shape))
		if err := decoder.Decode(valPtr); err != nil {
			return nil, errors.Wrap(err, "str")
		}
		goTensor := reflect.Indirect(reflect.ValueOf(valPtr)).Interface()
		val, err = tensorflow.NewTensor(goTensor)
		if err != nil {
			return nil, err
		}
	} else {
		var buf []byte
		if err := decoder.Decode(&buf); err != nil {
			return nil, errors.Wrap(err, "buf")
		}
		val, err = tensorflow.ReadTensor(dataType, shape, bytes.NewReader(buf))
		if err != nil {
			return nil, errors.Wrap(err, "val")
		}
	}
	return val, nil
}

// EncodeTensorMap encodes a map[string]*tensorflow.Tensor into a gob.Encoder.
// See DecodeTensorMap.
func EncodeTensorMap(encoder *gob.Encoder, m map[string]*tensorflow.Tensor) error {
	if err := encoder.Encode(uint32(len(m))); err != nil {
		return err
	}

	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	for _, key := range keys {
		val := m[key]
		if err := encoder.Encode(key); err != nil {
			return err
		}
		if err := EncodeTensor(encoder, val); err != nil {
			return err
		}
	}

	return nil
}

// DecodeTensorMap decodes a map[string]*tensorflow.Tensor from a gob.Decoder
// and returns it.  See EncodeTensorMap.
func DecodeTensorMap(decoder *gob.Decoder) (map[string]*tensorflow.Tensor, error) {
	m := map[string]*tensorflow.Tensor{}

	var numFeeds uint32
	if err := decoder.Decode(&numFeeds); err != nil {
		return nil, errors.Wrap(err, "numFeeds")
	}

	for i := uint32(0); i < numFeeds; i++ {
		var key string
		if err := decoder.Decode(&key); err != nil {
			return nil, errors.Wrap(err, "key")
		}
		val, err := DecodeTensor(decoder)
		if err != nil {
			return nil, err
		}
		m[key] = val
	}

	return m, nil
}
