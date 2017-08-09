package tf

import "testing"

func TestModelWeights(t *testing.T) {
	model := loadTestModel(t)
	weights, err := model.Weights()
	if err != nil {
		t.Fatal(err)
	}
	if len(weights) != len(model.TrainableVariables) {
		t.Fatalf("model.Weights() = %+v; len != %d", weights, len(model.TrainableVariables))
	}
	for output, weight := range weights {
		t.Logf("Weights()[%s:%d] = %+v", output.Op.Name(), output.Index, weight.Value())
	}
}
