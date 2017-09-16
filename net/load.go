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
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, errors.Errorf("expected status %+v; got %+v", http.StatusOK, resp.StatusCode)
	}
	return resp.Body, nil
}
