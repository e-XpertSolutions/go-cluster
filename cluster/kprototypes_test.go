package cluster

import (
	"reflect"
	"testing"
)

var cn1, cc1 *DenseMatrix

func initCent() {
	cn1 = NewDenseMatrix(2, 1, []float64{1, 1})
	cc1 = NewDenseMatrix(2, 1, []float64{1, 2})
}

func TestKPrototypes_FitModel(t *testing.T) {
	initMatrixKModes()
	initCentersKModes()
	initCent()

	tests := []struct {
		km         *KPrototypes
		X          *DenseMatrix
		CentersCat *DenseMatrix
		CentersNum *DenseMatrix
		wantErr    bool
	}{
		{km: &KPrototypes{DistanceFunc: HammingDistance, InitializationFunc: InitCao, CategoricalInd: []int{1}, Gamma: 1, ClustersNumber: 2, RunsNumber: 1, MaxIterationNumber: 10, WeightVectors: [][]float64{{1, 1, 1}}, ModelPath: "km.txt"},
			X:          m1,
			wantErr:    false,
			CentersCat: cc1,
			CentersNum: cn1,
		},
	}
	for i, tt := range tests {

		err := tt.km.FitModel(tt.X)
		if (err != nil) != tt.wantErr {
			t.Errorf("%d. KPrototypes.FitModel() error = %v, wantErr %v", i, err, tt.wantErr)
		}

		if err == nil {
			var wrong bool
			for i := 0; i < len(tt.km.ClusterCentroidsNum.RawMatrix().Data); i++ {
				if tt.km.ClusterCentroidsNum.RawMatrix().Data[i] != tt.CentersNum.Dense.RawMatrix().Data[i] {
					wrong = true

				}
			}
			if wrong {
				t.Errorf("%d. KPrototypes.ClusterCentroids = %v, want %v", i, tt.km.ClusterCentroidsNum.RawMatrix().Data, tt.CentersNum.Dense.RawMatrix().Data)
			}
		}

	}
}

func TestKPrototypes_Predict(t *testing.T) {
	initMatrixKModes()
	initCentersKModes()
	initCent()
	type args struct {
		X *DenseMatrix
	}
	tests := []struct {
		train   *DenseMatrix
		pred    *DenseMatrix
		km      *KPrototypes
		fit     bool
		want    *DenseVector
		wantErr bool
	}{
		{km: &KPrototypes{DistanceFunc: HammingDistance, InitializationFunc: InitCao, CategoricalInd: []int{1}, Gamma: 1, ClustersNumber: 2, RunsNumber: 1, MaxIterationNumber: 10, WeightVectors: [][]float64{{1, 1, 1}}, ModelPath: "km.txt"},
			train:   m1,
			fit:     true,
			pred:    NewDenseMatrix(1, 2, []float64{1, 1}),
			wantErr: false,
			want:    NewDenseVector(1, []float64{0})},

		{km: &KPrototypes{DistanceFunc: HammingDistance, InitializationFunc: InitCao, ClustersNumber: 2, CategoricalInd: []int{1}, Gamma: 1, RunsNumber: 1, MaxIterationNumber: 10, WeightVectors: [][]float64{{1, 1, 1}}, ModelPath: "km.txt"},
			train:   m1,
			pred:    NewDenseMatrix(1, 2, []float64{1, 1}),
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
