package cluster

import (
	"reflect"
	"sort"
	"testing"
)

var m1, m2 *DenseMatrix
var c1, c2 *DenseMatrix

func initMatrixKModes() {
	m1 = NewDenseMatrix(6, 2, []float64{1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2})
	m2 = NewDenseMatrix(7, 2, []float64{1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2})
}

func initCentersKModes() {
	c1 = NewDenseMatrix(2, 2, []float64{1, 1, 1, 2})
	c2 = NewDenseMatrix(2, 2, []float64{1, 1, 1, 2})
}

func TestKModes_FitModel(t *testing.T) {
	initMatrixKModes()
	initCentersKModes()

	tests := []struct {
		km      *KModes
		X       *DenseMatrix
		Centers *DenseMatrix
		wantErr bool
	}{
		{km: &KModes{DistanceFunc: HammingDistance, InitializationFunc: InitCao, ClustersNumber: 2, RunsNumber: 1, MaxIterationNumber: 10, WeightVectors: [][]float64{{1, 1, 1}}, ModelPath: "km.txt"},
			X:       m2,
			wantErr: false,
			Centers: c2,
		},

		{km: &KModes{DistanceFunc: HammingDistance, InitializationFunc: InitRandom, ClustersNumber: 2, RunsNumber: 1, MaxIterationNumber: 10, WeightVectors: [][]float64{{1, 1, 1}}, ModelPath: "km.txt"},
			X:       m1,
			wantErr: false,
			Centers: c1,
		},

		{km: &KModes{DistanceFunc: HammingDistance, InitializationFunc: InitCao, ClustersNumber: 0, RunsNumber: 1, MaxIterationNumber: 10, WeightVectors: [][]float64{{1, 1, 1}}, ModelPath: "km.txt"},
			X:       m1,
			wantErr: true,
			Centers: c1,
		},
		{km: &KModes{DistanceFunc: HammingDistance, InitializationFunc: InitCao, ClustersNumber: 5, RunsNumber: 1, MaxIterationNumber: 0, WeightVectors: [][]float64{{1, 1, 1}}, ModelPath: "km.txt"},
			X:       m1,
			wantErr: true,
			Centers: c1,
		},
		{km: &KModes{DistanceFunc: nil, InitializationFunc: InitCao, ClustersNumber: 5, RunsNumber: 1, MaxIterationNumber: 10, WeightVectors: [][]float64{{1, 1, 1}}, ModelPath: "km.txt"},
			X:       m1,
			wantErr: true,
			Centers: c1,
		},
		{km: &KModes{DistanceFunc: HammingDistance, InitializationFunc: nil, ClustersNumber: 5, RunsNumber: 1, MaxIterationNumber: 10, WeightVectors: [][]float64{{1, 1, 1}}, ModelPath: "km.txt"},
			X:       m1,
			wantErr: true,
			Centers: c1,
		},
	}
	for i, tt := range tests {

		err := tt.km.FitModel(tt.X)
		if (err != nil) != tt.wantErr {
			t.Errorf("%d. KModes.FitModel() error = %v, wantErr %v", i, err, tt.wantErr)
		}
		if err == nil {
			tt.km.ClusterCentroids = sortMatrix(tt.km.ClusterCentroids)

			var wrong bool
			for i := 0; i < len(tt.km.ClusterCentroids.RawMatrix().Data); i++ {
				if tt.km.ClusterCentroids.RawMatrix().Data[i] != tt.Centers.Dense.RawMatrix().Data[i] {
					wrong = true

				}
			}
			if wrong {
				t.Errorf("%d. KModes.ClusterCentroids = %v, want %v", i, tt.km.ClusterCentroids.RawMatrix().Data, tt.Centers.Dense.RawMatrix().Data)
			}
		}
	}

}

func TestKModes_SaveModel(t *testing.T) {
	tests := []struct {
		km      *KModes
		wantErr bool
	}{
		{km: &KModes{DistanceFunc: HammingDistance, InitializationFunc: InitCao, ClustersNumber: 2, RunsNumber: 1, MaxIterationNumber: 10, WeightVectors: [][]float64{{1, 1, 1}}, ModelPath: "km.txt"},
			wantErr: false},
		{km: &KModes{DistanceFunc: HammingDistance, InitializationFunc: InitCao, ClustersNumber: 2, RunsNumber: 1, MaxIterationNumber: 10, WeightVectors: [][]float64{{1, 1, 1}}, ModelPath: ""},
			wantErr: true},
	}
	for _, tt := range tests {

		if err := tt.km.SaveModel(); (err != nil) != tt.wantErr {
			t.Errorf("KModes.SaveModel() error = %v, wantErr %v", err, tt.wantErr)
		}

	}
}

func TestKModes_LoadModel(t *testing.T) {
	tests := []struct {
		name    string
		km      *KModes
		wantErr bool
	}{
		{km: &KModes{DistanceFunc: HammingDistance, InitializationFunc: InitCao, ClustersNumber: 2, RunsNumber: 1, MaxIterationNumber: 10, WeightVectors: [][]float64{{1, 1, 1}}, ModelPath: "km.txt"},
			wantErr: false},
		{km: &KModes{DistanceFunc: HammingDistance, InitializationFunc: InitCao, ClustersNumber: 2, RunsNumber: 1, MaxIterationNumber: 10, WeightVectors: [][]float64{{1, 1, 1}}, ModelPath: ""},
			wantErr: true},
	}
	for i, tt := range tests {
		if tt.km.ModelPath != "" {
			tt.km.SaveModel()
			newKM := &KModes{ModelPath: tt.km.ModelPath}
			if err := newKM.LoadModel(); (err != nil) != tt.wantErr {
				t.Errorf("%d. KModes.LoadModel() error = %v, wantErr %v", i, err, tt.wantErr)
			}
			if tt.km.ClustersNumber != newKM.ClustersNumber {
				t.Errorf("%d. KModes.LoadModel() loaded= %v, want = %v", i, newKM, tt.km)
			}
		} else {
			if err := tt.km.LoadModel(); (err != nil) != tt.wantErr {
				t.Errorf("%d. KModes.LoadModel() error = %v, wantErr %v", i, err, tt.wantErr)
			}
		}

	}
}

func TestKModes_Predict(t *testing.T) {
	initMatrixKModes()
	initCentersKModes()
	tests := []struct {
		train   *DenseMatrix
		pred    *DenseMatrix
		km      *KModes
		fit     bool
		want    *DenseVector
		wantErr bool
	}{
		{km: &KModes{DistanceFunc: HammingDistance, InitializationFunc: InitCao, ClustersNumber: 2, RunsNumber: 1, MaxIterationNumber: 10, WeightVectors: [][]float64{{1, 1, 1}}, ModelPath: "km.txt"},
			train:   m1,
			fit:     true,
			pred:    NewDenseMatrix(1, 2, []float64{1, 1}),
			wantErr: false,
			want:    NewDenseVector(1, []float64{0})},

		{km: &KModes{DistanceFunc: HammingDistance, InitializationFunc: InitCao, ClustersNumber: 2, RunsNumber: 1, MaxIterationNumber: 10, WeightVectors: [][]float64{{1, 1, 1}}, ModelPath: "km.txt"},
			train:   m1,
			pred:    NewDenseMatrix(1, 2, []float64{1, 1}),
			wantErr: true,
			want:    NewDenseVector(0, nil)},

		{km: &KModes{DistanceFunc: HammingDistance, InitializationFunc: InitCao, ClustersNumber: 2, RunsNumber: 1, MaxIterationNumber: 10, WeightVectors: [][]float64{{1, 1, 1}}, ModelPath: "km.txt"},
			train:   m1,
			fit:     true,
			pred:    NewDenseMatrix(1, 3, []float64{1, 1, 1}),
			wantErr: true,
			want:    NewDenseVector(0, nil)},
	}
	for _, tt := range tests {
		if tt.fit {
			tt.km.FitModel(tt.train)
		}
		got, err := tt.km.Predict(tt.pred)
		if (err != nil) != tt.wantErr {
			t.Errorf("KModes.Predict() error = %v, wantErr %v", err, tt.wantErr)
			return
		}
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("KModes.Predict() = %v, want %v", got, tt.want)
		}

	}
}

func Test_findHighestMapValue(t *testing.T) {

	tests := []struct {
		m     map[float64]float64
		want  float64
		want1 bool
	}{
		{m: map[float64]float64{1.0: 1.0, 2.0: 2.0}, want: 2.0, want1: false},
		{m: map[float64]float64{1.0: 5.7, 2.0: 2.0, 3.0: 5.8}, want: 3.0, want1: false},
		{m: map[float64]float64{}, want: 0, want1: true},
	}
	for _, tt := range tests {

		got, got1 := findHighestMapValue(tt.m)
		if got != tt.want {
			t.Errorf("findHighestMapValue() got = %v, want %v", got, tt.want)
		}
		if got1 != tt.want1 {
			t.Errorf("findHighestMapValue() got1 = %v, want %v", got1, tt.want1)
		}

	}
}

func sortMatrix(X *DenseMatrix) *DenseMatrix {

	var sorted [][]float64
	xr, xc := X.Dims()

	sorted = make([][]float64, xr)
	for i := 0; i < xr; i++ {
		sorted[i] = X.RawRowView(i)
	}

	sort.Slice(sorted, func(a, b int) bool {
		for x := range sorted[a] {
			if sorted[a][x] == sorted[b][x] {
				continue
			}
			return sorted[a][x] < sorted[b][x]
		}
		return false
	})

	flatten := make([]float64, 0)
	for _, el := range sorted {
		flatten = append(flatten, el...)
	}

	return NewDenseMatrix(xr, xc, flatten)

}
