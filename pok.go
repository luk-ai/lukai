package lukai

import (
	"log"
	"os"
	"sync"
	"time"

	lru "github.com/hashicorp/golang-lru"
	"github.com/pkg/errors"

	"github.com/luk-ai/lukai/debounce"
	"github.com/luk-ai/lukai/protobuf/clientpb"
	"github.com/luk-ai/lukai/tf"
)

var (
	ErrNotImplemented = errors.New("not implemented")
	PokEdgeAddress    = "localhost:5003"

	// ModelCacheSize controls how many training models are cached between
	// training iterations.
	ModelCacheSize = 3

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

	modelCache *lru.Cache
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

	var err error
	mt.modelCache, err = lru.NewWithEvict(ModelCacheSize, func(key, val interface{}) {
		val.(*tf.Model).Close()
	})
	if err != nil {
		return nil, err
	}

	if err := mt.loadExamplesMeta(); err != nil {
		return nil, err
	}

	return &mt, nil
}
