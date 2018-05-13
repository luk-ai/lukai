package tf

import (
	"bytes"
	"encoding/binary"
	"io"
	"reflect"
	"sort"

	"github.com/pkg/errors"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

func nDTensorType(t reflect.Type, n int) reflect.Type {
	for i := 0; i < n; i++ {
		t = reflect.SliceOf(t)
	}
	return t
}

func nDimensionalTensor(t reflect.Type, n int) interface{} {
	return reflect.New(nDTensorType(t, n)).Interface()
}

func DecodeStringND(t reflect.Type, shape []int64, r io.Reader) (reflect.Value, error) {
	if len(shape) == 0 {
		str, err := DecodeString(r)
		if err != nil {
			return reflect.Value{}, err
		}
		return reflect.ValueOf(str), nil
	}
	n := int(shape[0])
	childType := t.Elem()
	childShape := shape[1:]
	slice := reflect.MakeSlice(t, n, n)
	for i := 0; i < n; i++ {
		v, err := DecodeStringND(childType, childShape, r)
		if err != nil {
			return reflect.Value{}, err
		}
		slice.Index(i).Set(v)
	}
	return slice, nil
}

func EncodeStringND(w io.Writer, val reflect.Value) error {
	if val.Kind() == reflect.String {
		str := val.Interface().(string)
		if err := EncodeString(w, str); err != nil {
			return err
		}
		return nil
	}

	n := val.Len()
	for i := 0; i < n; i++ {
		if err := EncodeStringND(w, val.Index(i)); err != nil {
			return err
		}
	}
	return nil
}

// EncodeTensor encodes a tensor into a gob.Encoder. See DecodeTensor.
func EncodeTensor(w io.Writer, val *tensorflow.Tensor) error {
	dataType := val.DataType()
	if err := EncodeDataType(w, dataType); err != nil {
		return err
	}
	shape := val.Shape()
	if err := EncodeInt64Array(w, shape); err != nil {
		return err
	}

	if dataType == tensorflow.String {
		goVal := val.Value()
		if err := EncodeStringND(w, reflect.ValueOf(goVal)); err != nil {
			return err
		}
		return nil
	}

	var buf bytes.Buffer
	valLen, err := val.WriteContentsTo(&buf)
	if err != nil {
		return err
	}
	length := int64(buf.Len())
	if valLen != length {
		return errors.Errorf("expected tensor to write %d bytes, only wrote %d", valLen, length)
	}
	if err := binary.Write(w, binary.LittleEndian, length); err != nil {
		return err
	}
	n, err := buf.WriteTo(w)
	if err != nil {
		return err
	}
	if n != length {
		return errors.Errorf("expected to write %d bytes; only wrote %d", length, n)
	}
	return nil
}

// DecodeTensor decodes a tensor from a gob.Decoder and returns it.
// See EncodeTensor.
func DecodeTensor(r io.Reader) (*tensorflow.Tensor, error) {
	dataType, err := DecodeDataType(r)
	if err != nil {
		return nil, err
	}
	shape, err := DecodeInt64Array(r)
	if err != nil {
		return nil, errors.Wrap(err, "shape")
	}

	var val *tensorflow.Tensor
	if dataType == tensorflow.String {
		t := nDTensorType(reflect.TypeOf(""), len(shape))
		strND, err := DecodeStringND(t, shape, r)
		if err != nil {
			return nil, err
		}
		goTensor := reflect.Indirect(strND).Interface()
		val, err = tensorflow.NewTensor(goTensor)
		if err != nil {
			return nil, err
		}
		return val, nil
	} else {
		var length int64
		if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
			return nil, err
		}
		lr := io.LimitedReader{R: r, N: length}
		val, err = tensorflow.ReadTensor(dataType, shape, &lr)
		if err != nil {
			return nil, errors.Wrapf(err, "error reading (length %d, datatype %+v, shape %+v, remaining %d)", length, dataType, shape, lr.N)
		}
		if lr.N != 0 {
			return nil, errors.Errorf("only read %d bytes; wanted %d", length-lr.N, length)
		}
	}
	return val, nil
}

// EncodeTensorMap encodes a map[string]*tensorflow.Tensor into a gob.Encoder.
// See DecodeTensorMap.
func EncodeTensorMap(w io.Writer, m map[string]*tensorflow.Tensor) error {
	if err := binary.Write(w, binary.LittleEndian, int64(len(m))); err != nil {
		return err
	}

	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	for _, key := range keys {
		if err := EncodeString(w, key); err != nil {
			return err
		}
		val := m[key]
		if err := EncodeTensor(w, val); err != nil {
			return err
		}
	}

	return nil
}

// DecodeTensorMap decodes a map[string]*tensorflow.Tensor from a gob.Decoder
// and returns it.  See EncodeTensorMap.
func DecodeTensorMap(r io.Reader) (map[string]*tensorflow.Tensor, error) {
	m := map[string]*tensorflow.Tensor{}

	var numFeeds int64
	if err := binary.Read(r, binary.LittleEndian, &numFeeds); err != nil {
		return nil, errors.Wrap(err, "numFeeds")
	}

	for i := int64(0); i < numFeeds; i++ {
		key, err := DecodeString(r)
		if err != nil {
			return nil, err
		}
		val, err := DecodeTensor(r)
		if err != nil {
			return nil, errors.Wrapf(err, "tensor value, i %d, key %q", i, key)
		}
		m[key] = val
	}

	return m, nil
}

// Decode string reads a string from an reader.
// Format is: int64(len) + body
func DecodeString(r io.Reader) (string, error) {
	var strLen int64
	if err := binary.Read(r, binary.LittleEndian, &strLen); err != nil {
		return "", errors.Wrap(err, "length")
	}
	strBody := make([]byte, strLen)
	n, err := r.Read(strBody)
	if err != nil {
		return "", errors.Wrap(err, "body")
	}
	if int64(n) != strLen {
		return "", errors.Errorf("wanted to read %d bytes, only read %d", strLen, n)
	}
	return string(strBody), nil
}

// EncodeString encodes a string into the writer.
func EncodeString(w io.Writer, body string) error {
	if err := binary.Write(w, binary.LittleEndian, int64(len(body))); err != nil {
		return errors.Wrap(err, "length")
	}
	if _, err := w.Write([]byte(body)); err != nil {
		return errors.Wrap(err, "body")
	}
	return nil
}

// DecodeStringArray decodes a string array from the bytes.
// Format is: int64(num strings) + repeated (string)
func DecodeStringArray(r io.Reader) ([]string, error) {
	var count int64
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, err
	}

	var strs []string
	for i := int64(0); i < count; i++ {
		str, err := DecodeString(r)
		if err != nil {
			return nil, err
		}
		strs = append(strs, str)
	}
	return strs, nil
}

// EncodeStringArray decodes a string array from the bytes.
// Format is: int64(num strings) + repeated (string)
func EncodeStringArray(w io.Writer, arr []string) error {
	if err := binary.Write(w, binary.LittleEndian, int64(len(arr))); err != nil {
		return err
	}
	for _, v := range arr {
		if err := EncodeString(w, v); err != nil {
			return err
		}
	}
	return nil
}

// DecodeInt64Array decodes a int64 array from the bytes.
// Format is: int64(num) + repeated (int64)
func DecodeInt64Array(r io.Reader) ([]int64, error) {
	var count int64
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, err
	}

	var arr []int64
	for i := int64(0); i < count; i++ {
		var v int64
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		arr = append(arr, v)
	}
	return arr, nil
}

// EncodeInt64Array encodes a int64 array.
// Format is: int64(num) + repeated (int64)
func EncodeInt64Array(w io.Writer, arr []int64) error {
	if err := binary.Write(w, binary.LittleEndian, int64(len(arr))); err != nil {
		return err
	}
	for _, v := range arr {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	return nil
}

// EncodeDataType writes the data type to the writer.
func EncodeDataType(w io.Writer, dt tensorflow.DataType) error {
	if err := binary.Write(w, binary.LittleEndian, int64(dt)); err != nil {
		return err
	}
	return nil
}

// DecodeDataType decodes the data type from the reader.
func DecodeDataType(r io.Reader) (tensorflow.DataType, error) {
	var dt int64
	if err := binary.Read(r, binary.LittleEndian, &dt); err != nil {
		return 0, err
	}
	return tensorflow.DataType(dt), nil
}
