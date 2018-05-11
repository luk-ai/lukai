package metrics

import (
	"math"
	"testing"

	"github.com/luk-ai/lukai/protobuf/clientpb"
)

func TestMean(t *testing.T) {
	m := Get(clientpb.REDUCE_MEAN)
	m.Add(1.0)
	m.Add(2.0)
	m.Add(3.0)
	want := 2.0
	out := m.Val()
	if out != want {
		t.Errorf("got %f; want %f", out, want)
	}
}

func TestMax(t *testing.T) {
	m := Get(clientpb.REDUCE_MAX)
	m.Add(-1.0)
	m.Add(2.0)
	m.Add(3.0)
	want := 3.0
	out := m.Val()
	if out != want {
		t.Errorf("got %f; want %f", out, want)
	}
}

func TestMin(t *testing.T) {
	m := Get(clientpb.REDUCE_MIN)
	m.Add(-1.0)
	m.Add(2.0)
	m.Add(3.0)
	want := -1.0
	out := m.Val()
	if out != want {
		t.Errorf("got %f; want %f", out, want)
	}
}

func TestSum(t *testing.T) {
	m := Get(clientpb.REDUCE_SUM)
	m.Add(-1.0)
	m.Add(2.0)
	m.Add(3.0)
	want := 4.0
	out := m.Val()
	if out != want {
		t.Errorf("got %f; want %f", out, want)
	}
}

func TestProd(t *testing.T) {
	m := Get(clientpb.REDUCE_PROD)
	m.Add(-1.0)
	m.Add(2.0)
	m.Add(3.0)
	want := -6.0
	out := m.Val()
	if out != want {
		t.Errorf("got %f; want %f", out, want)
	}
}

func TestP(t *testing.T) {
	cases := []struct {
		m    clientpb.MetricReduce
		want float64
	}{
		{clientpb.REDUCE_P1, 1.0},
		{clientpb.REDUCE_P5, 5.0},
		{clientpb.REDUCE_P10, 10.0},
		{clientpb.REDUCE_P25, 25.0},
		{clientpb.REDUCE_P50, 50.0},
		{clientpb.REDUCE_P75, 75.0},
		{clientpb.REDUCE_P90, 90.0},
		{clientpb.REDUCE_P95, 95.0},
		{clientpb.REDUCE_P99, 99.0},
	}

	for i, c := range cases {
		m := Get(c.m)
		for i := float64(0); i < 100; i += 0.1 {
			m.Add(i)
		}
		out := m.Val()
		if math.Abs(out-c.want) > 0.5 {
			t.Errorf("%d. got %f; want %f", i, out, c.want)
		}
	}
}
