package cluster

import (
	"reflect"
	"sort"
	"testing"
)

var want1, want2, want3 [][]KV
var mat1, mat2, mat3, mat4 *DenseMatrix
var cen1, cen2 *DenseMatrix

func initVectorsInit() {
	mat1 = NewDenseMatrix(7, 2, []float64{1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2})
	mat2 = NewDenseMatrix(4, 1, []float64{1, 1, 2, 2})
	mat3 = NewDenseMatrix(6, 2, []float64{1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2})
	mat4 = NewDenseMatrix(10, 1, []float64{1, 2, 3, 4, 1, 2, 3, 3, 1, 2})

	cen2 = NewDenseMatrix(2, 2, []float64{1, 1, 1, 2})
	cen1 = NewDenseMatrix(2, 2, []float64{1, 1, 1, 2})

	want1 = make([][]KV, 2)
	want1[0] = []KV{KV{Key: 1, Value: 7}}
	want1[1] = []KV{KV{Key: 2, Value: 4}, KV{Key: 1, Value: 3}}

	want2 = make([][]KV, 1)
	want2[0] = []KV{KV{Key: 1, Value: 2}, KV{Key: 2, Value: 2}}

	want3 = make([][]KV, 1)
	want3[0] = []KV{KV{Key: 1, Value: 3}, KV{Key: 2, Value: 3}, KV{Key: 3, Value: 3}, KV{Key: 4, Value: 1}}
}

func TestCreateFrequencyTable(t *testing.T) {
	initVectorsInit()

	tests := []struct {
		X    *DenseMatrix
		want [][]KV
	}{
		{X: mat1, want: want1},
		{X: mat2, want: want2},
		{X: mat4, want: want3},
	}
	for _, tt := range tests {
		got := CreateFrequencyTable(tt.X)

		for i := range got {
			sort.Slice(got[i], func(a, b int) bool {
				if got[i][a].Value == got[i][b].Value {
					return got[i][a].Key < got[i][b].Key
				}
				return false
			})
		}

		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("CreateFrequencyTable() = %v, want %v", got, tt.want)
		}

	}
}

func TestInitHuang(t *testing.T) {
	initVectorsInit()
	type args struct {
		X              *DenseMatrix
		clustersNumber int
		distFunc       DistanceFunction
	}
	tests := []struct {
		args    args
		want    *DenseMatrix
		wantErr bool
	}{
		{args: args{X: mat1, clustersNumber: 2, distFunc: HammingDistance}, want: cen1, wantErr: false},
		{args: args{X: mat3, clustersNumber: 2, distFunc: HammingDistance}, want: cen2, wantErr: false},
	}
	for _, tt := range tests {

		got, err := InitHuang(tt.args.X, tt.args.clustersNumber, tt.args.distFunc)
		got = sortMatrix(got)
		if (err != nil) != tt.wantErr {
			t.Errorf("InitHuang() error = %v, wantErr %v", err, tt.wantErr)
			return
		}
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("InitHuang() = %v, want %v", got.Dense, tt.want.Dense)
		}

	}
}
