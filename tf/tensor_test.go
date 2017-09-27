package tf

import (
	"bytes"
	"reflect"
	"testing"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

func TestEncodeDecodeTensor(t *testing.T) {
	want := [][]float64{
		{0, 1, 2, 3, 4, 5, 6, 8, 9},
		{20, 21, 22, 23, 24, 25, 27, 28, 29},
	}
	tensor, err := tensorflow.NewTensor(want)
	if err != nil {
		t.Fatalf("%+v", err)
	}

	var buf bytes.Buffer
	if err := EncodeTensor(&buf, tensor); err != nil {
		t.Fatalf("%+v", err)
	}
	tensor2, err := DecodeTensor(&buf)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	out := tensor2.Value()
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("DecodeTensor(EncodeTensor(%+v)) = %+v", want, out)
	}
}

func TestEncodeDecodeStringArray(t *testing.T) {
	want := []string{
		"hello this is some string",
		"another string",
		"last string",
	}

	var buf bytes.Buffer
	if err := EncodeStringArray(&buf, want); err != nil {
		t.Fatalf("%+v", err)
	}
	out, err := DecodeStringArray(&buf)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("DecodeTensor(EncodeTensor(%+v)) = %+v", want, out)
	}
}
