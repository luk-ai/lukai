package pok

import (
	"bytes"
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

type exampleVal struct {
	feeds   map[string]interface{}
	fetches []string
	targets []string
}

// exampleToVal returns an example struct with the tensorflow.Tensor items
// replaced with the go value. For use with reflect.DeepEqual.
func exampleToVal(ex example) exampleVal {
	v := exampleVal{
		fetches: ex.fetches,
		targets: ex.targets,
		feeds:   map[string]interface{}{},
	}
	for k, tensor := range ex.feeds {
		v.feeds[k] = tensor.Value()
	}
	return v
}

// newTensor creates a new tensor and panics if an error is returned.
// Only use for test code!
func newTensor(v interface{}) *tensorflow.Tensor {
	tensor, err := tensorflow.NewTensor(v)
	if err != nil {
		panic(err)
	}
	return tensor
}

func TestExamplesSerialization(t *testing.T) {
	/*
			All TF types

		  TF_FLOAT = 1,
		  TF_DOUBLE = 2,
		  TF_INT32 = 3,
		  TF_UINT8 = 4,
		  TF_INT16 = 5,
		  TF_INT8 = 6,
		  TF_STRING = 7,
		  TF_COMPLEX64 = 8,
		  TF_COMPLEX = 8,
		  TF_INT64 = 9,
		  TF_BOOL = 10,
		  TF_QINT8 = 11,
		  TF_QUINT8 = 12,
		  TF_QINT32 = 13,
		  TF_BFLOAT16 = 14,
		  TF_QINT16 = 15,
		  TF_QUINT16 = 16,
		  TF_UINT16 = 17,
		  TF_COMPLEX128 = 18,
		  TF_HALF = 19,
		  TF_RESOURCE = 20,
		  TF_VARIANT = 21,

	*/
	cases := []example{
		{
			feeds: map[string]*tensorflow.Tensor{
				"float32": newTensor(float32(100)),
			},
			targets: []string{"a", "b"},
			fetches: []string{"c:0", "d:1"},
		},
		{
			feeds: map[string]*tensorflow.Tensor{
				"str":  newTensor("str"),
				"str2": newTensor("str2"),
			},
		},
		{
			feeds: map[string]*tensorflow.Tensor{
				"int8":    newTensor(int8(100)),
				"uint8":   newTensor(uint8(100)),
				"int16":   newTensor(int16(100)),
				"uint16":  newTensor(uint16(100)),
				"int32":   newTensor(int32(100)),
				"int64":   newTensor(int64(100)),
				"float32": newTensor(float32(100)),
				"float64": newTensor(float64(100)),
			},
		},
		{
			feeds: map[string]*tensorflow.Tensor{
				"true":  newTensor(true),
				"false": newTensor(false),
			},
		},
		{
			feeds: map[string]*tensorflow.Tensor{
				"complex128": newTensor(complex(10, 10)),
			},
		},
		{
			feeds: map[string]*tensorflow.Tensor{
				"[]float32": newTensor([]float32{1, 2, 3, 4}),
				"[][]float32": newTensor([][]float32{
					{1, 2, 3, 4},
					{5, 6, 7, 8},
				}),
				"[][][]float32": newTensor([][][]float32{
					{
						{1, 2, 3, 4},
						{5, 6, 7, 8},
					},
					{
						{11, 12, 13, 14},
						{15, 16, 17, 18},
					},
				}),
			},
		},
		{
			feeds: map[string]*tensorflow.Tensor{
				"[]string": newTensor([]string{"a", "b", "c"}),
				"[][]string": newTensor([][]string{
					{"a", "b", "c"},
					{"d", "e", "f"},
				}),
				"[][][]string": newTensor([][][]string{
					{
						{"a", "b", "c"},
						{"d", "e", "f"},
					},
					{
						{"a", "b", "c"},
						{"d", "e", "f"},
					},
				}),
			},
		},
	}

	var buf bytes.Buffer
	var buf2 bytes.Buffer
	var out example
	for i, c := range cases {
		buf.Reset()
		buf2.Reset()

		inVal := exampleToVal(c)
		{
			n, err := c.writeTo(&buf)
			if err != nil {
				t.Errorf("%d. writeTo: %+v", i, err)
			}
			if n != buf.Len() {
				t.Errorf("%d. writeTo returned wrong length; got %d; want %d", i, buf.Len(), n)
			}
		}

		bufBytes := buf.String()
		{
			bufLen := buf.Len()
			n, err := out.readFrom(&buf)
			if err != nil {
				t.Errorf("%d. readFrom: %+v", i, err)
			}
			if n != bufLen {
				t.Errorf("%d. readFrom returned wrong length; got %d; want %d", i, bufLen, n)
			}
			outVal := exampleToVal(out)
			if !reflect.DeepEqual(outVal, inVal) {
				t.Errorf("%d. reloaded example val = %+v; not %+v", i, outVal, inVal)
			}
		}

		{
			n, err := out.writeTo(&buf2)
			if err != nil {
				t.Errorf("%d. writeTo: %+v", i, err)
			}
			if n != buf2.Len() {
				t.Errorf("%d. writeTo returned wrong length; got %d; want %d", i, buf2.Len(), n)
			}
			buf2Bytes := buf2.String()
			if bufBytes != buf2Bytes {
				t.Errorf(
					"%d. reserialized bytes don't equal original, len(buf) = %d, len(buf2) = %d\n%q\n%q",
					i, len(bufBytes), len(buf2Bytes), bufBytes, buf2Bytes,
				)
			}
		}
	}
}

func TestLog(t *testing.T) {
	dir, err := ioutil.TempDir("", "pok-server-TestLog")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer os.RemoveAll(dir)
	mt := ModelType{
		Domain:    "test",
		ModelType: "test",
		DataDir:   dir,
	}
	mt.examplesMeta.saveIndex = func() {}

	examples := []example{
		{
			feeds: map[string]*tensorflow.Tensor{
				"tensor": newTensor(int64(0)),
			},
			targets: []string{"target"},
		},
		{
			feeds: map[string]*tensorflow.Tensor{
				"tensor": newTensor(int64(1)),
			},
			targets: []string{"target"},
		},
		{
			feeds: map[string]*tensorflow.Tensor{
				"tensor": newTensor(int64(2)),
			},
			targets: []string{"target"},
		},
	}
	for _, example := range examples {
		if err := mt.Log(example.feeds, example.targets); err != nil {
			t.Fatalf("%+v", err)
		}
	}
	seen := map[int64]struct{}{}
	count := 0
	for i := 0; i < 1000; i++ {
		examples, err := mt.getNExamples(int64(len(examples)))
		if err != nil {
			t.Fatalf("%+v", err)
		}
		count += len(examples)
		for _, ex := range examples {
			val := ex.feeds["tensor"].Value().(int64)
			seen[val] = struct{}{}

			out := exampleToVal(ex)
			want := exampleToVal(examples[val])
			if !reflect.DeepEqual(out, want) {
				t.Fatalf("examples[%d] = %+v; != %+v", val, want, out)
			}
		}
		if len(seen) == len(examples) {
			break
		}
	}
	if len(seen) != len(examples) {
		t.Fatalf(
			"failed to get all examples from getNExamples after %d examples. Seen: %+v",
			count, seen,
		)
	}
}
