package net

import (
	"io"
	"net/http"

	"github.com/pkg/errors"
)

// OpenModel opens the model for reading.
func OpenModel(url string) (io.ReadCloser, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, errors.Wrap(err, url)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, errors.Errorf("%q: expected status %+v; got %+v", url, http.StatusOK, resp.StatusCode)
	}
	return resp.Body, nil
}
