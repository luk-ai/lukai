package lukai

import (
	"io/ioutil"
	"net"
	"os"
	"strings"
	"sync"
	"testing"

	"go.uber.org/goleak"
	"golang.org/x/net/context"

	"google.golang.org/grpc"

	"github.com/pkg/errors"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"

	"github.com/luk-ai/lukai/protobuf/aggregatorpb"
	"github.com/luk-ai/lukai/testutil"
)

type testEdgeServer struct {
	aggregatorpb.EdgeServer
	aggregatorpb.AggregatorServer

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
	aggregatorpb.RegisterAggregatorServer(grpcServer, s)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		wg.Done()
		if err := grpcServer.Serve(lis); err != nil {
			t.Log(err)
		}
	}()
	wg.Wait()
	return s
}

func (s *testEdgeServer) stop() {
	s.grpc.GracefulStop()
}

func (s *testEdgeServer) ProdModel(ctx context.Context, in *aggregatorpb.ProdModelRequest) (*aggregatorpb.ProdModelResponse, error) {
	return nil, ErrNotImplemented
}

func (s *testEdgeServer) GetWork(
	req *aggregatorpb.GetWorkRequest,
	stream aggregatorpb.Aggregator_GetWorkServer,
) error {
	return ErrNotImplemented
}

func (s *testEdgeServer) ReportWork(stream aggregatorpb.Aggregator_ReportWorkServer) error {
	return ErrNotImplemented
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

	ctx := context.Background()
	mt.StartTraining(ctx)
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

	mt.StartTraining(ctx)
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

	if err := mt.ExamplesError(); err != nil && !strings.Contains(err.Error(), "context canceled") {
		t.Error(err)
	}

	if err := mt.TrainingError(); err != nil && !strings.Contains(err.Error(), "context canceled") {
		t.Error(err)
	}
}
