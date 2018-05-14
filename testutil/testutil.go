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
	fWrap := func() error {
		// Fast fail if the test has already failed.
		if t.Failed() {
			return nil
		}
		return f()
	}
	if err := backoff.Retry(fWrap, opts); err != nil {
		t.Fatal(err)
	}
}
