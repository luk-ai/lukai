package tf

import (
	"math"
	"testing"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

func TestModelWeights(t *testing.T) {
	model := loadTestModel(t)
	weights, err := model.Weights()
	if err != nil {
		t.Fatal(err)
	}

	if len(weights) != len(model.TrainableVariables) {
		t.Fatalf("model.Weights() = %+v; len != %d", weights, len(model.TrainableVariables))
	}
}

func TestAddScaledWeights(t *testing.T) {
	model := loadTestModel(t)
	weights, err := model.Weights()
	if err != nil {
		t.Fatal(err)
	}
	var1 := weights[0].Value().([][]float32)

	scale, err := tensorflow.NewTensor(0.5)
	if err != nil {
		t.Fatal(err)
	}
	if err := model.AddScaledWeights(weights, scale); err != nil {
		t.Fatal(err)
	}

	weights2, err := model.Weights()
	if err != nil {
		t.Fatal(err)
	}
	var2 := weights2[0].Value().([][]float32)

	for i, row := range var1 {
		for j, v := range row {
			want := float64(v * 1.5)
			out := float64(var2[i][j])
			if math.Abs(want-out) > 0.000001 {
				t.Errorf("weights2[0][%d][%d] = %f; not %f", i, j, out, want)
			}
		}
	}
}
