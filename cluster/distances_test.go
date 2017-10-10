package cluster

import (
	"reflect"
	"testing"
)

var mw1, mw2, mw3 *DenseMatrix
var a, b, c, d *DenseVector
var w1, w2 []float64

func initMWdist() {
	mw1 = NewDenseMatrix(3, 1, []float64{1, 1, 1})
	mw2 = NewDenseMatrix(3, 2, []float64{1, 2, 2, 1, 1, 2})
	mw3 = NewDenseMatrix(3, 2, []float64{1, 2, 1, 1, 1, 2})
}

func initVectorsdist() {
	a = NewDenseVector(3, []float64{1, 1, 1})
	b = NewDenseVector(3, []float64{1, 1, 1})
	c = NewDenseVector(4, []float64{1, 2, 3, 4})
	d = NewDenseVector(4, []float64{1, 1, 3, 3})
	w1 = []float64{1, 1, 1, 2}
	w2 = []float64{1}
}

func TestHammingDistance(t *testing.T) {

	initVectorsdist()
	type args struct {
		v1 *DenseVector
		v2 *DenseVector
	}
	tests := []struct {
		args    args
		want    float64
		wantErr bool
	}{
		{args: args{v1: a, v2: b}, want: 0, wantErr: false},
		{args: args{v1: a, v2: c}, want: -1, wantErr: true},
		{args: args{v1: c, v2: d}, want: 2, wantErr: false},
	}
	for i, tt := range tests {

		got, err := HammingDistance(tt.args.v1, tt.args.v2)
		if (err != nil) != tt.wantErr {
			t.Errorf("%d. HammingDistance() error = %v, wantErr %v", i, err, tt.wantErr)
			return
		}
		if got != tt.want {
			t.Errorf("%d. HammingDistance() = %v, want %v", i, got, tt.want)
		}

	}
}

func TestWeightedHammingDistance(t *testing.T) {
	initVectorsdist()
	type args struct {
		v1 *DenseVector
		v2 *DenseVector
	}
	tests := []struct {
		args         args
		want         float64
		wantErr      bool
		weightVector []float64
	}{
		{args: args{v1: c, v2: d}, want: 3, wantErr: false, weightVector: w1},
		{args: args{v1: a, v2: d}, want: -1, wantErr: true, weightVector: w1},
		{args: args{v1: c, v2: d}, want: -1, wantErr: true, weightVector: w2},
	}
	for i, tt := range tests {
		SetWeights(tt.weightVector)

		got, err := WeightedHammingDistance(tt.args.v1, tt.args.v2)
		if (err != nil) != tt.wantErr {
			t.Errorf("%d. WeightedHammingDistance() error = %v, wantErr %v", i, err, tt.wantErr)
			return
		}
		if got != tt.want {
			t.Errorf("%d. WeightedHammingDistance() = %v, want %v", i, got, tt.want)
		}
	}
}

func TestEuclideanDistance(t *testing.T) {
	initVectorsdist()
	type args struct {
		v1 *DenseVector
		v2 *DenseVector
	}
	tests := []struct {
		args    args
		want    float64
		wantErr bool
	}{
		{args: args{v1: a, v2: b}, want: 0, wantErr: false},
		{args: args{v1: a, v2: c}, want: -1, wantErr: true},
		{args: args{v1: c, v2: d}, want: 1.41421, wantErr: false},
	}
	for _, tt := range tests {

		got, err := EuclideanDistance(tt.args.v1, tt.args.v2)
		got = float64(int(got*100000)) / 100000
		if (err != nil) != tt.wantErr {
			t.Errorf("EuclideanDistance() error = %v, wantErr %v", err, tt.wantErr)
			return
		}
		if got != tt.want {
			t.Errorf("EuclideanDistance() = %v, want %v", got, tt.want)
		}

	}
}

func Test_maxVal(t *testing.T) {

	tests := []struct {
		table []float64
		want  float64
	}{
		{table: []float64{1.1, 1.0, 2.1, 5.0}, want: 5.0},
		{table: []float64{1.0, 1.0, 0.9}, want: 1.0},
	}
	for _, tt := range tests {

		if got := maxVal(tt.table); got != tt.want {
			t.Errorf("maxVal() = %v, want %v", got, tt.want)
		}

	}
}

func TestComputeWeights(t *testing.T) {
	initMWdist()
	type args struct {
		X   *DenseMatrix
		imp float64
	}
	tests := []struct {
		args args
		want []float64
	}{
		{args: args{X: mw1, imp: 1}, want: []float64{1}},
		{args: args{X: mw2, imp: 1}, want: []float64{1, 1}},
		{args: args{X: mw3, imp: 1}, want: []float64{0, 1}},
	}
	for _, tt := range tests {

		if got := ComputeWeights(tt.args.X, tt.args.imp); !reflect.DeepEqual(got, tt.want) {
			t.Errorf("ComputeWeights() = %v, want %v", got, tt.want)
		}

	}
}
