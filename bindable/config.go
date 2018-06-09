package bindable

import (
	"time"

	"github.com/luk-ai/lukai"
	"github.com/luk-ai/lukai/units"
)

// SetMaxConcurrentTrainingJobs sets the maximum number of training jobs that
// can be running. One ModelType will only ever train on one, but it's possible
// for multiple instances of ModelType to train multiple at the same time.
func SetMaxConcurrentTrainingJobs(n int) {
	lukai.SetMaxConcurrentTrainingJobs(n)
}

// SetMaxFileSize sets how large each example file can be.
func SetMaxFileSize(bytes int) {
	lukai.MaxFileSize = units.Bytes(bytes) * units.B
}

// SetMaxDiskUsage sets how much disk can be used for examples.
func SetMaxDiskUsage(bytes int) {
	lukai.MaxDiskUsage = units.Bytes(bytes) * units.B
}

// SetMaxFileDuration sets the duration worth of examples that will be stored in
// one file. Specified in seconds.
func SetMaxFileDuration(seconds int) {
	lukai.MaxFileDuration = time.Duration(seconds) * time.Second
}

// SetMaxFileRetention sets the duration worth of examples that will be kept.
func SetMaxFileRetention(seconds int) {
	lukai.MaxFileRetention = time.Duration(seconds) * time.Second
}
