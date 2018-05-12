package lukai

import (
	"io/ioutil"
	"net"
	"os"
	"testing"

	"go.uber.org/goleak"
	"golang.org/x/net/context"

	"google.golang.org/grpc"

	"github.com/pkg/errors"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"

	"github.com/luk-ai/lukai/protobuf/aggregatorpb"
	"github.com/luk-ai/lukai/testutil"
)

var errNotImplemented = errors.New("not implemented")

type testEdgeServer struct {
	grpc *grpc.Server
}

func newTestEdgeServer(t *testing.T) *testEdgeServer {
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatal(err)
	}
	EdgeAddress = lis.Addr().String()

	grpcServer := grpc.NewServer()
	s := &testEdgeServer{
		grpc: grpcServer,
	}
	aggregatorpb.RegisterEdgeServer(grpcServer, s)
	go func() {
		if err := grpcServer.Serve(lis); err != nil {
			t.Log(err)
		}
	}()
	return s
}

func (s *testEdgeServer) stop() {
	s.grpc.GracefulStop()
}

func (s *testEdgeServer) ProdModel(ctx context.Context, in *aggregatorpb.ProdModelRequest) (*aggregatorpb.ProdModelResponse, error) {
	return nil, errNotImplemented
}

func (s *testEdgeServer) GetWork(
	req *aggregatorpb.GetWorkRequest,
	stream aggregatorpb.Edge_GetWorkServer,
) error {
	return errNotImplemented
}

func (s *testEdgeServer) ReportWork(stream aggregatorpb.Edge_ReportWorkServer) error {
	return errNotImplemented
}

func newTestModelType(t *testing.T) (*ModelType, func()) {
	dir, err := ioutil.TempDir("", "pok-server-TestLog")
	if err != nil {
		t.Fatalf("%+v", err)
	}

	mt, err := MakeModelType("test", "test", dir)
	if err != nil {
		t.Fatal(err)
	}

	return mt, func() {
		if err := mt.Close(); err != nil {
			t.Error(err)
		}
		os.RemoveAll(dir)
	}
}

func TestModelTraining(t *testing.T) {
	defer goleak.VerifyNoLeaks(t)

	s := newTestEdgeServer(t)
	defer s.stop()

	mt, stop := newTestModelType(t)
	defer stop()

	mt.StartTraining()
	mt.training.Lock()
	if mt.training.running {
		t.Fatalf("ModelType shouldn't be training due to no examples")
	}
	mt.training.Unlock()

	if err := mt.Log(map[string]*tensorflow.Tensor{
		"Placeholder_1": newTensor([]float32{1, 2, 3}),
	}, []string{"train"}); err != nil {
		t.Fatal(err)
	}

	mt.StartTraining()
	mt.training.Lock()
	if !mt.training.running {
		t.Fatalf("ModelType should be training")
	}
	mt.training.Unlock()

	mt.StopTraining()
	testutil.SucceedsSoon(t, func() error {
		mt.training.Lock()
		defer mt.training.Unlock()

		if mt.training.running {
			return errors.New("ModelType should stop training")
		}
		return nil
	})
}
