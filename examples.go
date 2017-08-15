package pok

import (
	"bytes"
	"encoding/gob"
	"io"
	"math/rand"
	"os"
	"path"
	"sort"
	"time"

	"github.com/pkg/errors"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"

	"github.com/d4l3k/pok/units"
)

var (
	// FileSize is the target size an example file will be.
	FileSize = 1 * units.MB
	// MaxFileRetention is the number of days worth of examples kept.
	MaxFileRetention = 14 * units.Day
	// MaxFileSize is the number of bytes that will used for examples.
	MaxFileSize = 50 * units.MB
)

type example struct {
	feeds   map[string]*tensorflow.Tensor
	fetches []string
	targets []string
}

type writeCounter struct {
	n      int
	target io.Writer
}

func (c *writeCounter) Write(p []byte) (int, error) {
	n, err := c.target.Write(p)
	if err != nil {
		return 0, err
	}
	c.n += n
	return n, nil
}

func (ex example) writeTo(w io.Writer) (int, error) {
	c := writeCounter{target: w}
	encoder := gob.NewEncoder(&c)
	if err := encoder.Encode(uint32(len(ex.feeds))); err != nil {
		return 0, err
	}

	for key, val := range ex.feeds {
		if err := encoder.Encode(key); err != nil {
			return 0, err
		}
		if err := encoder.Encode(val.DataType()); err != nil {
			return 0, err
		}
		if err := encoder.Encode(val.Shape()); err != nil {
			return 0, err
		}
		if val.DataType() == tensorflow.String {
			if err := encoder.Encode(val.Value()); err != nil {
				return 0, err
			}
		} else {
			var buf bytes.Buffer
			if _, err := val.WriteContentsTo(&buf); err != nil {
				return 0, err
			}
			if err := encoder.Encode(buf.Bytes()); err != nil {
				return 0, err
			}
		}
	}

	if err := encoder.Encode(ex.fetches); err != nil {
		return 0, err
	}

	if err := encoder.Encode(ex.targets); err != nil {
		return 0, err
	}

	return c.n, nil
}

type readCounter struct {
	n      int
	target io.Reader
}

func (c *readCounter) Read(p []byte) (int, error) {
	n, err := c.target.Read(p)
	if err != nil {
		return 0, err
	}
	c.n += n
	return n, nil
}

func (ex *example) readFrom(r io.Reader) (int, error) {
	*ex = example{
		feeds: map[string]*tensorflow.Tensor{},
	}

	c := readCounter{target: r}
	decoder := gob.NewDecoder(&c)

	var numFeeds uint32
	if err := decoder.Decode(&numFeeds); err != nil {
		return 0, errors.Wrap(err, "numFeeds")
	}

	for i := uint32(0); i < numFeeds; i++ {
		var key string
		if err := decoder.Decode(&key); err != nil {
			return 0, errors.Wrap(err, "key")
		}
		var dataType tensorflow.DataType
		if err := decoder.Decode(&dataType); err != nil {
			return 0, errors.Wrap(err, "dataType")
		}
		var shape []int64
		if err := decoder.Decode(&shape); err != nil {
			return 0, errors.Wrap(err, "shape")
		}

		var val *tensorflow.Tensor
		var err error
		if dataType == tensorflow.String {
			var str string
			if err := decoder.Decode(&str); err != nil {
				return 0, errors.Wrap(err, "str")
			}
			val, err = tensorflow.NewTensor(str)
			if err != nil {
				return 0, err
			}
		} else {
			var buf []byte
			if err := decoder.Decode(&buf); err != nil {
				return 0, errors.Wrap(err, "buf")
			}
			val, err = tensorflow.ReadTensor(dataType, shape, bytes.NewReader(buf))
			if err != nil {
				return 0, errors.Wrap(err, "val")
			}
		}
		ex.feeds[key] = val
	}

	if err := decoder.Decode(&ex.fetches); err != nil {
		return 0, errors.Wrap(err, "fetches")
	}

	if err := decoder.Decode(&ex.targets); err != nil {
		return 0, errors.Wrap(err, "targets")
	}
	return c.n, nil
}

type exampleFile struct {
	name             string
	modified         time.Time
	examplePositions []int32
}

type exampleIndex struct {
	totalExamples int64
	files         []exampleFile
}

// Log records model input->output pairs for later use in training. This data is
// saved locally only.
// - feeds key is the tensorflow output and should be in the form "name:output#".
// - targets is the name of the tensorflow target and should be in the form "name".
func (mt *ModelType) Log(feeds map[string]*tensorflow.Tensor, targets []string) error {
	return ErrNotImplemented
}

func (mt *ModelType) getNExamples(n int64) ([]example, error) {
	mt.examplesMeta.RLock()
	defer mt.examplesMeta.RUnlock()

	fileReads := map[string][]int64{}

	for i := int64(0); i < n; i++ {
		exampleIndex := rand.Int63n(mt.examplesMeta.index.totalExamples)
		seenCount := int64(0)
		for _, file := range mt.examplesMeta.index.files {
			seenSoFar := seenCount + int64(len(file.examplePositions))
			if exampleIndex < seenSoFar {
				fileReads[file.name] = append(fileReads[file.name], exampleIndex-seenCount)
				break
			}
			seenCount = seenSoFar
		}
		if seenCount == mt.examplesMeta.index.totalExamples {
			return nil, errors.Errorf("failed to find file for example index %d", exampleIndex)
		}
	}

	examples := make([]example, n)
	i := 0
	for filename, offsets := range fileReads {
		// Sort the offsets to improve disk read performance.
		sort.Slice(offsets, func(i, j int) bool {
			return offsets[i] < offsets[j]
		})

		f, err := os.OpenFile(path.Join(mt.DataDir, filename), os.O_RDONLY, 0700)
		if err != nil {
			return nil, err
		}

		for _, offset := range offsets {
			if _, err := f.Seek(offset, 0); err != nil {
				return nil, err
			}
			if _, err := examples[i].readFrom(f); err != nil {
				return nil, err
			}
			i++
		}
	}

	return examples, nil
}
