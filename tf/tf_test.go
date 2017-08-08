package tf

import (
	"bytes"
	"log"
	"math"
	"os"
	"testing"
)

func TestExtractNodeName(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", ""},
		{"asdf", "asdf"},
		{"test/foo/bar:10", "test/foo/bar"},
	}

	for i, c := range cases {
		out := ExtractNodeName(c.in)
		if out != c.want {
			t.Errorf("%d. ExtractNodeName(%q) = %q; not %q", i, c.in, out, c.want)
		}
	}
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
}
