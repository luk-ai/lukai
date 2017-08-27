package net

import (
	"bytes"
	"errors"
	"io"
	"reflect"
	"sync"
	"testing"

	"github.com/d4l3k/pok/protobuf/aggregatorpb"
	"github.com/d4l3k/pok/testutil"
	"github.com/d4l3k/pok/units"
)

func TestModelWeightsWriter(t *testing.T) {

	chunkChan := make(chan *aggregatorpb.ModelWeightChunk)

	var expect bytes.Buffer
	w := NewModelWeightsWriter(func(chunk aggregatorpb.ModelWeightChunk) error {
		chunk.Body = append([]byte{}, chunk.Body...)
		chunkChan <- &chunk
		return nil
	})
	r := ReadModelWeights(func() (*aggregatorpb.ModelWeightChunk, error) {
		return <-chunkChan, nil
	})

	var buf bytes.Buffer
	mu := struct {
		sync.Mutex

		done bool
	}{}
	go func() {
		if _, err := buf.ReadFrom(r); err != nil {
			t.Fatal(err)
		}
		mu.Lock()
		defer mu.Unlock()
		mu.done = true
	}()

	mw := io.MultiWriter(w, &expect)

	mw.Write([]byte(`test 1234 ajksdfklajsdfklasdkjflasdf`))
	mw.Write(make([]byte, 1*units.MB))
	mw.Write([]byte(`last part of the message`))

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	testutil.SucceedsSoon(t, func() error {
		mu.Lock()
		defer mu.Unlock()

		if !mu.done {
			return errors.New("ReadFrom hasn't finished")
		}
		return nil
	})

	want := expect.Bytes()
	out := buf.Bytes()
	if !reflect.DeepEqual(want, out) {
		t.Fatalf("got '%s' (len %d); want '%s' (len %d)", out, len(out), want, len(want))
	}
}
