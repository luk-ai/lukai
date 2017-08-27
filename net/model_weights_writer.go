package net

import (
	"io"

	"github.com/pkg/errors"

	"github.com/d4l3k/pok/protobuf/aggregatorpb"
	"github.com/d4l3k/pok/units"
)

const ModelWeightChunkSize = 64 * units.KB

func ReadModelWeights(recv func() (*aggregatorpb.ModelWeightChunk, error)) io.ReadCloser {
	r, w := io.Pipe()
	go func() {
		for {
			resp, err := recv()
			if err != nil {
				w.CloseWithError(err)
				return
			}
			if resp == nil {
				w.CloseWithError(errors.New("received something other than weight chunk"))
				return
			}
			w.Write(resp.Body)
			if !resp.More {
				w.Close()
				return
			}
		}
	}()
	return r
}

// ModelWeightsWriter writes ModelWeightChunks.
type ModelWeightsWriter struct {
	send func(aggregatorpb.ModelWeightChunk) error
	buf  [ModelWeightChunkSize]byte
	size int
}

func (w *ModelWeightsWriter) Write(b []byte) (int, error) {
	bLen := len(b)
	for len(b) > 0 {
		numRead := ModelWeightChunkSize - w.size
		if numRead > len(b) {
			numRead = len(b)
		}
		copy(w.buf[w.size:], b[:numRead])
		w.size += numRead
		b = b[numRead:]

		if w.size == ModelWeightChunkSize {
			if err := w.sendChunk(true); err != nil {
				return 0, err
			}
		}
	}
	return bLen, nil
}

func (w *ModelWeightsWriter) sendChunk(more bool) error {
	body := make([]byte, w.size)
	copy(body, w.buf[:w.size])
	if err := w.send(aggregatorpb.ModelWeightChunk{
		Body: body,
		More: more,
	}); err != nil {
		return errors.Wrapf(err, "sendChunk failed: more = %b", more)
	}
	w.size = 0
	return nil
}

func (w *ModelWeightsWriter) Close() error {
	if w.size > 0 {
		if err := w.sendChunk(false); err != nil {
			return err
		}
	}
	return nil
}

func NewModelWeightsWriter(send func(aggregatorpb.ModelWeightChunk) error) *ModelWeightsWriter {
	return &ModelWeightsWriter{
		send: send,
	}
}
