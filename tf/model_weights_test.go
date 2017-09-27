package tf

import (
	"bytes"
	"math"
	"testing"
)

func TestModelWeights(t *testing.T) {
	model := loadTestModel(t)
	weights, err := model.weights()
	if err != nil {
		t.Fatal(err)
	}

	if len(weights) != len(model.Meta.TrainableVariables) {
		t.Fatalf("model.Weights() = %+v; len != %d", weights, len(model.Meta.TrainableVariables))
	}
}

func TestAddScaledWeights(t *testing.T) {
	model := loadTestModel(t)
	weights, err := model.weights()
	if err != nil {
		t.Fatal(err)
	}
	var1 := weights[0].Value().([][]float32)

	weightsMap, err := model.WeightsMap()
	if err != nil {
		t.Fatal(err)
	}

	if err := model.AddWeights(0.5, weightsMap); err != nil {
		t.Fatal(err)
	}

	weights2, err := model.weights()
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

func TestImportExportWeights(t *testing.T) {
	model := loadTestModel(t)
	var buf bytes.Buffer
	if err := model.ExportWeights(&buf); err != nil {
		t.Fatalf("%+v", err)
	}

	t.Logf("buf len = %d", buf.Len())
	weights, err := LoadWeights(&buf)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if err := model.SetWeights(weights); err != nil {
		t.Fatalf("%+v", err)
	}
}
