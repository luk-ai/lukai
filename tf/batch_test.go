package tf

import (
	"reflect"
	"testing"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

func makeTensorN(n int, val float32) *tensorflow.Tensor {
	arr := make([]float32, n)
	for i := range arr {
		arr[i] = val
	}
	tensor, err := tensorflow.NewTensor([][]float32{arr})
	if err != nil {
		panic(err)
	}
	return tensor
}

func TestBatcher(t *testing.T) {
	t.Parallel()

	l := 2
	n := 3
	batcher, err := NewTensorBatcher(n, tensorflow.Float, []int64{-1, int64(l)})
	if err != nil {
		t.Fatal(err)
	}
	session, err := batcher.Session()
	if err != nil {
		t.Fatal(err)
	}
	out, err := batcher.Batch(session, []*tensorflow.Tensor{
		makeTensorN(l, 1.0),
		makeTensorN(l, 2.0),
		makeTensorN(l, 3.0),
	})
	if err != nil {
		t.Fatal(err)
	}
	val := out.Value().([][]float32)
	want := [][]float32{
		{1.0, 1.0},
		{2.0, 2.0},
		{3.0, 3.0},
	}
	if !reflect.DeepEqual(val, want) {
		t.Errorf("batcher.Batch(...) = %+v; not %+v", val, want)
	}
}
