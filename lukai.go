package lukai

import (
	"context"
	"log"
	"os"
	"sync"
	"time"

	lru "github.com/hashicorp/golang-lru"
	"github.com/pkg/errors"
	"golang.org/x/time/rate"

	"github.com/luk-ai/lukai/debounce"
	"github.com/luk-ai/lukai/protobuf/aggregatorpb"
	"github.com/luk-ai/lukai/protobuf/clientpb"
	"github.com/luk-ai/lukai/tf"
)

var (
	ErrNotImplemented = errors.New("not implemented")
	EdgeAddress       = "dns://edge.luk.ai"

	// ModelCacheSize controls how many training models are cached between
	// training iterations.
	ModelCacheSize = 3

	DialTimeout           = 60 * time.Second
	outOfDateModelTimeout = 24 * time.Hour

	// ErrorRateLimit controls how often the client should report errors to the
	// server.
	ErrorRateLimit = 1 * time.Minute
	// MaxQueuedErrors controls how many errors can be queued before they start
	// getting discarded.
	MaxQueuedErrors = 10
)

type ModelType struct {
	Domain, ModelType, DataDir string

	prod struct {
		sync.RWMutex

		modelID    aggregatorpb.ModelID
		model      *tf.Model
		cache      tfOpCache
		lastUpdate time.Time
	}

	training struct {
		sync.Mutex

		running bool
		stop    context.CancelFunc

		err error
	}

	examplesMeta struct {
		sync.RWMutex

		index     clientpb.ExampleIndex
		saveIndex func()
		stop      func()

		err error
	}

	modelCache *lru.Cache

	errorLimiter *rate.Limiter
	errors       struct {
		sync.Mutex

		errors []aggregatorpb.Error
	}
}

// MakeModelType creates a new model type with a specified domain and model type
// and stores all training data in dataDir.
func MakeModelType(domain, modelType, dataDir string) (*ModelType, error) {
	mt := ModelType{
		Domain:       domain,
		ModelType:    modelType,
		DataDir:      dataDir,
		errorLimiter: rate.NewLimiter(rate.Every(ErrorRateLimit), 1),
	}

	if domain == "" {
		return nil, errors.Errorf("domain required")
	}
	if modelType == "" {
		return nil, errors.Errorf("modelType required")
	}
	if dataDir == "" {
		return nil, errors.Errorf("dataDir required")
	}

	if err := os.MkdirAll(dataDir, DirPerm); err != nil {
		return nil, err
	}

	mt.examplesMeta.saveIndex, mt.examplesMeta.stop = debounce.Debounce(
		300*time.Millisecond,
		func() {
			if err := mt.saveExamplesMeta(); err != nil {
				log.Printf("saveExamplesMeta error: %+v", err)

				mt.examplesMeta.Lock()
				defer mt.examplesMeta.Unlock()

				mt.examplesMeta.err = err
				return
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

func (mt *ModelType) Close() error {
	mt.examplesMeta.stop()
	if err := mt.saveExamplesMeta(); err != nil {
		return err
	}
	return nil
}
