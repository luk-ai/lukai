package pok

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"path"
	"reflect"
	"sort"
	"time"

	"github.com/gogo/protobuf/proto"
	"github.com/pkg/errors"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"

	"github.com/d4l3k/pok/protobuf/clientpb"
	"github.com/d4l3k/pok/units"
)

var (
	// MaxFileSize is the target size an example file will be.
	MaxFileSize = 1 * units.MB
	// MaxFileRetention is the number of days worth of examples kept.
	MaxFileRetention = 14 * units.Day
	// MaxDiskUsage is the number of bytes that will used for examples.
	MaxDiskUsage = 50 * units.MB
	// MaxDaysInFile is the number of days worth of examples that will be stored
	// in one day.
	MaxDaysInFile = 1 * units.Day
	// IndexFileName is the name of the examples index file.
	IndexFileName = "index.pb"
	// FilePerm is the file permission all the example files use.
	FilePerm os.FileMode = 0600
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

	keys := make([]string, 0, len(ex.feeds))
	for key := range ex.feeds {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	for _, key := range keys {
		val := ex.feeds[key]
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

func nDimensionalTensor(t reflect.Type, n int) interface{} {
	for i := 0; i < n; i++ {
		t = reflect.SliceOf(t)
	}
	return reflect.New(t).Interface()
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
			valPtr := nDimensionalTensor(reflect.TypeOf(""), len(shape))
			if err := decoder.Decode(valPtr); err != nil {
				return 0, errors.Wrap(err, "str")
			}
			goTensor := reflect.Indirect(reflect.ValueOf(valPtr)).Interface()
			val, err = tensorflow.NewTensor(goTensor)
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

// Log records model input->output pairs for later use in training. This data is
// saved locally only.
// - feeds key is the tensorflow output and should be in the form "name:output#".
// - targets is the name of the tensorflow target and should be in the form "name".
func (mt *ModelType) Log(feeds map[string]*tensorflow.Tensor, targets []string) error {
	mt.examplesMeta.Lock()
	defer mt.examplesMeta.Unlock()

	mt.ensureFilePresentLocked()
	mt.examplesMeta.index.TotalExamples += 1

	file := &mt.examplesMeta.index.Files[len(mt.examplesMeta.index.Files)-1]

	filePath := mt.filePath(file.Name)
	f, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE|os.O_APPEND, FilePerm)
	if err != nil {
		return err
	}

	offset := file.TotalSize
	if _, err := f.Seek(offset, 0); err != nil {
		return err
	}

	ex := example{
		feeds:   feeds,
		targets: targets,
	}
	n, err := ex.writeTo(f)
	if err != nil {
		return err
	}
	file.Positions = append(file.Positions, int32(offset))
	file.TotalSize += int64(n)
	mt.examplesMeta.index.TotalSize += int64(n)

	return nil
}

// ensureFilePresentLocked checks if there is a valid index entry, and if not,
// creates one.
func (mt *ModelType) ensureFilePresentLocked() {
	lastFile := &mt.examplesMeta.index.Files[len(mt.examplesMeta.index.Files)-1]

	// Create a new file if the file is too old.
	outdatedFile := lastFile.Created.Before(time.Now().Add(-MaxDaysInFile))
	// Create a new file if the last one is over the maximum file size.
	largeFile := lastFile.TotalSize >= int64(MaxFileSize)
	if !outdatedFile && !largeFile {
		return
	}

	now := time.Now()
	file := clientpb.ExampleFile{
		Name:    fmt.Sprintf("examples-%s", now.Format(time.RFC3339Nano)),
		Created: now,
	}
	mt.examplesMeta.index.Files = append(mt.examplesMeta.index.Files, file)
}

// saveExamplesMeta saves the examples index.
func (mt *ModelType) saveExamplesMeta() error {
	mt.examplesMeta.RLock()
	defer mt.examplesMeta.RUnlock()

	bytes, err := proto.Marshal(&mt.examplesMeta.index)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(mt.filePath(IndexFileName), bytes, FilePerm); err != nil {
		return err
	}

	return nil
}

// loadExamplesMeta loads the examples index.
func (mt *ModelType) loadExamplesMeta() error {
	mt.examplesMeta.Lock()
	defer mt.examplesMeta.Unlock()

	bytes, err := ioutil.ReadFile(mt.filePath(IndexFileName))
	if err == os.ErrNotExist {
		return nil
	} else if err != nil {
		return err
	}
	if err := proto.Unmarshal(bytes, &mt.examplesMeta.index); err != nil {
		return err
	}

	return nil
}

func (mt *ModelType) filePath(file string) string {
	return path.Join(mt.DataDir, file)
}

func (mt *ModelType) getNExamples(n int64) ([]example, error) {
	mt.examplesMeta.RLock()
	defer mt.examplesMeta.RUnlock()

	fileReads := map[string][]int64{}

	for i := int64(0); i < n; i++ {
		exampleIndex := rand.Int63n(mt.examplesMeta.index.TotalExamples)
		seenCount := int64(0)
		for _, file := range mt.examplesMeta.index.Files {
			seenSoFar := seenCount + int64(len(file.Positions))
			if exampleIndex < seenSoFar {
				fileReads[file.Name] = append(fileReads[file.Name], exampleIndex-seenCount)
				break
			}
			seenCount = seenSoFar
		}
		if seenCount == mt.examplesMeta.index.TotalExamples {
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

		f, err := os.OpenFile(mt.filePath(filename), os.O_RDONLY, FilePerm)
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
