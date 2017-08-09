package tf

import (
	"bytes"
	"log"
	"math"
	"os"
	"reflect"
	"testing"
)

func loadTestModel(t *testing.T) *Model {
	file, err := os.OpenFile("../testdata/model.tar.gz", os.O_RDONLY, 0755)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	model, err := LoadModel(file)
	if err != nil {
		t.Fatal(err)
	}

	return model
}

func TestLoadSaveModel(t *testing.T) {
	file, err := os.OpenFile("../testdata/model.tar.gz", os.O_RDONLY, 0755)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	model, err := LoadModel(file)
	if err != nil {
		t.Fatal(err)
	}

	if len(model.TrainableVariables) != 2 {
		t.Errorf("model.TrainableVariables = %+v; len should be 2", model.TrainableVariables)
	}

	var buf bytes.Buffer
	if err := model.Save(&buf); err != nil {
		t.Fatalf("%+v", err)
	}
	fi, err := file.Stat()
	if err != nil {
		t.Fatal(err)
	}
	a := float64(buf.Len())
	b := float64(fi.Size())
	if math.Abs(a-b) > math.Max(a, b)*0.2 {
		log.Fatalf("model sizes differ! model.Save() size = %d; original size = %d", buf.Len(), fi.Size())
	}

	model2, err := LoadModel(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(model, model2) {
		t.Fatalf("LoadModel(%+v.Save()) != %+v", model2, model)
	}
}
