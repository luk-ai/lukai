package testutil

import (
	"testing"
	"time"

	"github.com/cenkalti/backoff"
)

func SucceedsSoon(t *testing.T, f func() error) {
	opts := backoff.NewExponentialBackOff()
	opts.MaxElapsedTime = 15 * time.Second
	opts.InitialInterval = 1 * time.Microsecond
	if err := backoff.Retry(f, opts); err != nil {
		t.Fatal(err)
	}
}
