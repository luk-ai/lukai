package pok

import (
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"path"
	"sort"
	"time"

	"github.com/gogo/protobuf/proto"
	"github.com/pkg/errors"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"

	"github.com/d4l3k/pok/protobuf/clientpb"
	"github.com/d4l3k/pok/tf"
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
	FilePerm os.FileMode = tf.FilePerm
	DirPerm  os.FileMode = 0700
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
	gzw := gzip.NewWriter(&c)
	defer gzw.Close()

	encoder := gob.NewEncoder(gzw)

	if err := tf.EncodeTensorMap(encoder, ex.feeds); err != nil {
		return 0, err
	}

	if err := encoder.Encode(ex.fetches); err != nil {
		return 0, err
	}

	if err := encoder.Encode(ex.targets); err != nil {
		return 0, err
	}

	if err := gzw.Close(); err != nil {
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
	*ex = example{}

	c := readCounter{target: r}
	gzr, err := gzip.NewReader(&c)
	if err != nil {
		return 0, err
	}
	defer gzr.Close()
	decoder := gob.NewDecoder(gzr)

	ex.feeds, err = tf.DecodeTensorMap(decoder)
	if err != nil {
		return 0, errors.Wrap(err, "feeds")
	}

	if err := decoder.Decode(&ex.fetches); err != nil {
		return 0, errors.Wrap(err, "fetches")
	}

	if err := decoder.Decode(&ex.targets); err != nil {
		return 0, errors.Wrap(err, "targets")
	}
	return c.n, nil
}

// TotalExamples returns the number of examples that are currently saved
// locally.
func (mt *ModelType) TotalExamples() int64 {
	mt.examplesMeta.Lock()
	defer mt.examplesMeta.Unlock()

	return mt.examplesMeta.index.TotalExamples
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
	defer f.Close()

	ex := example{
		feeds:   feeds,
		targets: targets,
	}
	n, err := ex.writeTo(f)
	if err != nil {
		return err
	}
	file.Positions = append(file.Positions, int32(file.TotalSize))
	file.TotalSize += int64(n)
	mt.examplesMeta.index.TotalSize += int64(n)

	mt.examplesMeta.saveIndex()

	return nil
}

// ensureFilePresentLocked checks if there is a valid index entry, and if not,
// creates one.
func (mt *ModelType) ensureFilePresentLocked() {
	numFiles := len(mt.examplesMeta.index.Files)
	if numFiles > 0 {
		lastFile := &mt.examplesMeta.index.Files[numFiles-1]

		// Create a new file if the file is too old.
		outdatedFile := lastFile.Created.Before(time.Now().Add(-MaxDaysInFile))
		// Create a new file if the last one is over the maximum file size.
		largeFile := lastFile.TotalSize >= int64(MaxFileSize)
		if !outdatedFile && !largeFile {
			return
		}
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
	if os.IsNotExist(err) {
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

// int64Slice attaches the methods of Interface to []int64, sorting in increasing order.
type int64Slice []int64

func (p int64Slice) Len() int           { return len(p) }
func (p int64Slice) Less(i, j int) bool { return p[i] < p[j] }
func (p int64Slice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func (mt *ModelType) getNExamples(n int64) ([]example, error) {
	mt.examplesMeta.RLock()
	defer mt.examplesMeta.RUnlock()

	fileReads := map[string][]int64{}

	for i := int64(0); i < n; i++ {
		totalExamples := mt.examplesMeta.index.TotalExamples
		if totalExamples == 0 {
			break
		}

		exampleIndex := rand.Int63n(totalExamples)
		seenCount := int64(0)
		for _, file := range mt.examplesMeta.index.Files {
			seenSoFar := seenCount + int64(len(file.Positions))
			if exampleIndex < seenSoFar {
				fileReads[file.Name] = append(fileReads[file.Name], int64(file.Positions[exampleIndex-seenCount]))
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
		sort.Sort(int64Slice(offsets))

		f, err := os.OpenFile(mt.filePath(filename), os.O_RDONLY, FilePerm)
		if err != nil {
			return nil, err
		}
		defer f.Close()

		for _, offset := range offsets {
			if _, err := f.Seek(offset, 0); err != nil {
				return nil, err
			}
			if _, err := examples[i].readFrom(f); err != nil {
				return nil, errors.Wrapf(err, "failed to read from %q, offset %d", filename, offset)
			}
			i++
		}
	}

	return examples, nil
}

// getExampleBatch returns a batched set of n examples.
func (mt *ModelType) getExampleBatch(
	batchers batcherCache, cache tfOpCache, n int64,
) ([]example, error) {
	examples, err := mt.getNExamples(n)
	if err != nil {
		return nil, err
	}
	if n == 1 {
		return examples, nil
	}
	if int64(len(examples)) < n {
		log.Printf("don't have enough examples for batching; need %d; have %d", n, len(examples))
		return examples, nil
	}

	ex := example{
		feeds: map[string]*tensorflow.Tensor{},
	}

	values := map[string][]*tensorflow.Tensor{}
	for _, example := range examples {
		for name, val := range example.feeds {
			_, ok := batchers[name]
			if !ok {
				feed, err := cache.resolveFeed(name)
				if err != nil {
					return nil, err
				}
				shape, err := feed.Shape().ToSlice()
				if err != nil {
					return nil, err
				}
				batchers[name], err = tf.NewTensorBatcher(int(n), val.DataType(), shape)
				if err != nil {
					return nil, errors.Wrapf(err, "feed name %q", name)
				}
			}
			values[name] = append(values[name], val)
		}
		ex.fetches = example.fetches
		ex.targets = example.targets
	}

	for feed, values := range values {
		batcher := batchers[feed]
		out, err := batcher.Batch(values)
		if err != nil {
			return nil, errors.Wrapf(err, "feed %q", feed)
		}
		ex.feeds[feed] = out
	}
	return []example{ex}, nil
}

type batcherCache map[string]*tf.Batcher

func (c batcherCache) Close() error {
	for _, batcher := range c {
		if batcher == nil {
			continue
		}
		if err := batcher.Close(); err != nil {
			return err
		}
	}
	return nil
}
