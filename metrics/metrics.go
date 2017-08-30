package metrics

import (
	"math"

	"github.com/luk-ai/lukai/protobuf/clientpb"
	"github.com/streadway/quantile"
)

type Metric interface {
	Add(v float64)
	Val() float64
}

func Get(m clientpb.MetricReduce) Metric {
	switch m {
	case clientpb.REDUCE_MEAN:
		return NewMean()
	case clientpb.REDUCE_MIN:
		return NewMin()
	case clientpb.REDUCE_MAX:
		return NewMax()
	case clientpb.REDUCE_SUM:
		return NewSum()
	case clientpb.REDUCE_PROD:
		return NewProd()
	case clientpb.REDUCE_P1:
		return NewP(0.01)
	case clientpb.REDUCE_P5:
		return NewP(0.05)
	case clientpb.REDUCE_P10:
		return NewP(0.10)
	case clientpb.REDUCE_P25:
		return NewP(0.25)
	case clientpb.REDUCE_P50:
		return NewP(0.50)
	case clientpb.REDUCE_P75:
		return NewP(0.75)
	case clientpb.REDUCE_P90:
		return NewP(0.90)
	case clientpb.REDUCE_P95:
		return NewP(0.95)
	case clientpb.REDUCE_P99:
		return NewP(0.99)
	default:
		return NewUnknown()
	}
}

type Unknown struct{}

func NewUnknown() *Unknown {
	return &Unknown{}
}
func (*Unknown) Add(float64)  {}
func (*Unknown) Val() float64 { return -1 }

type Mean struct {
	sum float64
	n   int
}

func NewMean() *Mean {
	return &Mean{}
}

func (m *Mean) Add(v float64) {
	m.sum += v
	m.n += 1
}

func (m *Mean) Val() float64 {
	return m.sum / float64(m.n)
}

type Min struct {
	min float64
}

func NewMin() *Min {
	return &Min{
		min: math.MaxFloat64,
	}
}

func (m *Min) Add(v float64) {
	if v < m.min {
		m.min = v
	}
}

func (m *Min) Val() float64 {
	return m.min
}

type Max struct {
	max float64
}

func NewMax() *Max {
	return &Max{
		max: -math.MaxFloat64,
	}
}

func (m *Max) Add(v float64) {
	if v > m.max {
		m.max = v
	}
}

func (m *Max) Val() float64 {
	return m.max
}

type Sum struct {
	sum float64
}

func NewSum() *Sum {
	return &Sum{}
}

func (m *Sum) Add(v float64) {
	m.sum += v
}

func (m *Sum) Val() float64 {
	return m.sum
}

type Prod struct {
	prod float64
}

func NewProd() *Prod {
	return &Prod{
		prod: 1.0,
	}
}

func (m *Prod) Add(v float64) {
	m.prod *= v
}

func (m *Prod) Val() float64 {
	return m.prod
}

type P struct {
	p float64
	q *quantile.Estimator
}

func NewP(p float64) *P {
	return &P{
		p: p,
		q: quantile.New(quantile.Known(p, 0.005)),
	}
}

func (m *P) Add(v float64) {
	m.q.Add(v)
}

func (m *P) Val() float64 {
	return m.q.Get(m.p)
}
