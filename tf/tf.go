package tf

import (
	"strconv"
	"strings"

	tensorflowpb "github.com/d4l3k/pok/protobuf/tensorflow"
	"github.com/pkg/errors"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

type Model struct {
	Graph              *tensorflow.Graph
	Session            *tensorflow.Session
	SaverDef           tensorflowpb.SaverDef
	TrainableVariables []string
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
