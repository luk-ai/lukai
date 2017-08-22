package pok

import (
	"log"
	"os"
	"sync"
	"time"

	"github.com/pkg/errors"

	"github.com/d4l3k/pok/debounce"
	"github.com/d4l3k/pok/protobuf/clientpb"
	"github.com/d4l3k/pok/tf"
)

var (
	ErrNotImplemented = errors.New("not implemented")
	PokEdgeAddress    = "localhost:5003"

	outOfDateModelTimeout = 24 * time.Hour
)

type ModelType struct {
	Domain, ModelType, DataDir string

	prod struct {
		sync.RWMutex

		model      *tf.Model
		cache      tfOpCache
		lastUpdate time.Time
	}

	training struct {
		sync.Mutex

		running bool
		stop    chan struct{}
	}

	examplesMeta struct {
		sync.RWMutex

		index     clientpb.ExampleIndex
		saveIndex func()
		stop      func()
	}
}

func MakeModelType(domain, modelType, dataDir string) (*ModelType, error) {
	mt := ModelType{
		Domain:    domain,
		ModelType: modelType,
		DataDir:   dataDir,
	}

	if err := os.MkdirAll(dataDir, DirPerm); err != nil {
		return nil, err
	}

	mt.training.stop = make(chan struct{}, 1)
	mt.examplesMeta.saveIndex, mt.examplesMeta.stop = debounce.Debounce(
		300*time.Millisecond,
		func() {
			if err := mt.saveExamplesMeta(); err != nil {
				// TODO(d4l3k): Better error handling.
				log.Printf("saveExamplesMeta error: %+v", err)
			}
		},
	)

	if err := mt.loadExamplesMeta(); err != nil {
		return nil, err
	}

	return &mt, nil
}
