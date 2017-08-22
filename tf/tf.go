package tf

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/d4l3k/pok/protobuf/clientpb"
	tensorflowpb "github.com/d4l3k/pok/protobuf/tensorflow"
	"github.com/pkg/errors"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

type Model struct {
	Graph    *tensorflow.Graph
	Session  *tensorflow.Session
	SaverDef tensorflowpb.SaverDef
	Meta     clientpb.ModelMeta
	Prefix   string
}

func (model Model) ApplyPrefix(op string) string {
	if len(model.Prefix) > 0 {
		return model.Prefix + "/" + op
	}
	return op
}

// ParseNodeOutput returns the node name when given a "<name>:<output #>" pair.
func ParseNodeOutput(path string) (string, int, error) {
	parts := strings.Split(path, ":")
	if len(parts) > 2 {
		return "", 0, errors.Errorf("need 1-2 parts, got %d", len(parts))
	}
	if len(parts) == 2 {
		n, err := strconv.Atoi(parts[1])
		if err != nil {
			return "", 0, errors.Wrapf(err, "failed to parse second part of tensor")
		}
		return parts[0], n, nil
	}
	return parts[0], -1, nil
}

func (m *Model) Operation(path string) (*tensorflow.Operation, error) {
	name, _, err := ParseNodeOutput(path)
	if err != nil {
		return nil, err
	}
	op := m.Graph.Operation(name)
	if op == nil {
		return nil, errors.Errorf("invalid operation %q", name)
	}
	return op, nil
}

func (m *Model) Output(path string) (tensorflow.Output, error) {
	name, n, err := ParseNodeOutput(path)
	if err != nil {
		return tensorflow.Output{}, err
	}
	op := m.Graph.Operation(name)
	if op == nil {
		return tensorflow.Output{}, errors.Errorf("invalid operation %q", name)
	}
	return op.Output(n), nil
}

// GetModel fetches a model from a URL and loads it.
func GetModel(url string) (*Model, error) {
	req, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer req.Body.Close()
	model, err := LoadModel(req.Body)
	if err != nil {
		return nil, err
	}
	return model, nil
}

// Close closes the model.
func (model *Model) Close() error {
	return model.Session.Close()
}
