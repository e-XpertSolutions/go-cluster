package cluster

import (
	"encoding/gob"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"gonum.org/v1/gonum/mat"
)

// KPrototypes is a basic class for the k-prototypes algorithm, it contains all
// necessary information as alg. parameters, labels, centroids, ...
type KPrototypes struct {
	DistanceFunc        DistanceFunction
	InitializationFunc  InitializationFunction
	CategoricalInd      []int
	ClustersNumber      int
	RunsNumber          int
	MaxIterationNumber  int
	WeightVectors       [][]float64
	FrequencyTable      [][]map[float64]float64 // frequency table - list of lists with dictionaries containing frequencies of values per cluster and attribute
	MembershipNumTable  [][]float64             // membership table for numeric attributes - list of labels for each cluster
	LabelsCounter       []int
	Labels              *DenseVector
	ClusterCentroids    *DenseMatrix
	ClusterCentroidsCat *DenseMatrix
	ClusterCentroidsNum *DenseMatrix
	Gamma               float64
	IsFitted            bool
	ModelPath           string
}

// NewKPrototypes implements constructor for the KPrototypes struct.
func NewKPrototypes(dist DistanceFunction, init InitializationFunction, categorical []int, clusters int, runs int, iters int, weights [][]float64, g float64, modelPath string) *KPrototypes {
	rand.Seed(time.Now().UnixNano())
	return &KPrototypes{DistanceFunc: dist,
		InitializationFunc:  init,
		ClustersNumber:      clusters,
		CategoricalInd:      categorical,
		RunsNumber:          runs,
		MaxIterationNumber:  iters,
		Gamma:               g,
		WeightVectors:       weights,
		ModelPath:           modelPath,
		Labels:              &DenseVector{VecDense: new(mat.VecDense)},
		ClusterCentroidsCat: &DenseMatrix{Dense: new(mat.Dense)},
		ClusterCentroidsNum: &DenseMatrix{Dense: new(mat.Dense)},
	}
}

// FitModel main algorithm function which finds the best clusters centers for
// the given dataset X.
func (km *KPrototypes) FitModel(X *DenseMatrix) error {

	err := km.validateParameters()
	if err != nil {
		return fmt.Errorf("kmodes: failed to fit the model: %v", err)
	}
	xRows, xCols := X.Dims()

	// Partition data on two sets - one with categorical, other with numerical
	// data.
	xCat, xNum := km.partitionData(xRows, xCols, X)

	_, xCatCols := xCat.Dims()
	_, xNumCols := xNum.Dims()

	// Normalize numerical values.
	xNum = normalizeNum(xNum)

	// Initialize weightVector.
	SetWeights(km.WeightVectors[0])

	// Initialize clusters for categorical data.
	km.ClusterCentroidsCat, err = km.InitializationFunc(xCat, km.ClustersNumber, km.DistanceFunc)
	if err != nil {
		return fmt.Errorf("kmodes: failed to fit the model: %v", err)
	}

	// Initialize clusters for numerical data.
	km.ClusterCentroidsNum, err = InitNum(xNum, km.ClustersNumber, km.DistanceFunc)
	if err != nil {
		return fmt.Errorf("kmodes: failed to initialiaze cluster centers for numerical data: %v", err)
	}

	// Initialize labels vector
	km.Labels = NewDenseVector(xRows, nil)
	km.LabelsCounter = make([]int, km.ClustersNumber)

	// Create frequency table for categorical data.
	km.FrequencyTable = make([][]map[float64]float64, km.ClustersNumber)
	for i := range km.FrequencyTable {
		km.FrequencyTable[i] = make([]map[float64]float64, xCatCols)
		for j := range km.FrequencyTable[i] {
			km.FrequencyTable[i][j] = make(map[float64]float64)
		}
	}

	// Create membership table.
	km.MembershipNumTable = make([][]float64, km.ClustersNumber)
	for i := range km.MembershipNumTable {
		km.MembershipNumTable[i] = make([]float64, 0, 100)
	}

	// Perform initial assignements to clusters - in order to fill in frequency
	// table.
	for i := 0; i < xRows; i++ {
		rowCat := xCat.RowView(i).(*DenseVector)
		rowNum := xNum.RowView(i).(*DenseVector)
		newLabel, _, err := km.near(i, rowCat, rowNum)
		km.Labels.SetVec(i, newLabel)
		km.LabelsCounter[int(newLabel)]++
		if err != nil {
			return fmt.Errorf("kmodes: initial labels assignement failure: %v", err)
		}
		for j := 0; j < xCatCols; j++ {
			km.FrequencyTable[int(newLabel)][j][rowCat.At(j, 0)]++
		}
		km.MembershipNumTable[int(newLabel)] = append(km.MembershipNumTable[int(newLabel)], float64(i))

	}

	// Perform initial centers update - because iteration() starts with label
	// assignements.
	for i := 0; i < km.ClustersNumber; i++ {
		// Find new values for clusters centers.
		km.findNewCenters(xCatCols, xNumCols, i, xNum)

	}
	for i := 0; i < km.MaxIterationNumber; i++ {
		_, change, err := km.iteration(xNum, xCat)
		if err != nil {
			return fmt.Errorf("KMeans error at iteration %d: %v", i, err)
		}
		if change == false {
			km.IsFitted = true
			return nil
		}
	}

	return nil
}

func (km *KPrototypes) partitionData(xRows, xCols int, X *DenseMatrix) (*DenseMatrix, *DenseMatrix) {
	xCat := NewDenseMatrix(xRows, len(km.CategoricalInd), nil)
	xNum := NewDenseMatrix(xRows, xCols-len(km.CategoricalInd), nil)
	var lastCat, lastNum int
	for i := 0; i < xCols; i++ {
		vec := make([]float64, xRows)
		vec = mat.Col(vec, i, X)

		if km.CategoricalInd[lastCat] == i {
			xCat.SetCol(lastCat, vec)
			lastCat++
			if lastCat >= len(km.CategoricalInd) {
				lastCat--
			}
		} else {
			xNum.SetCol(lastNum, vec)
			lastNum++
		}
	}

	return xCat, xNum
}

func (km *KPrototypes) iteration(xNum, xCat *DenseMatrix) (float64, bool, error) {
	changed := make([]bool, km.ClustersNumber)
	var change bool
	var numOfChanges float64
	var totalCost float64

	for i := 0; i < km.ClustersNumber; i++ {
		km.MembershipNumTable[i] = nil
	}

	// Find closest cluster for all data vectors - assign new labels.
	xRowsNum, xNumCols := xNum.Dims()
	_, xColsCat := xCat.Dims()

	for i := 0; i < xRowsNum; i++ {
		rowCat := xCat.RowView(i).(*DenseVector)
		rowNum := xNum.RowView(i).(*DenseVector)
		newLabel, cost, err := km.near(i, rowCat, rowNum)
		if err != nil {
			return totalCost, change, fmt.Errorf("iteration error: %v", err)
		}
		totalCost += cost

		km.MembershipNumTable[int(newLabel)] = append(km.MembershipNumTable[int(newLabel)], float64(i))

		if newLabel != km.Labels.At(i, 0) {

			km.LabelsCounter[int(newLabel)]++
			km.LabelsCounter[int(km.Labels.At(i, 0))]--

			// Make changes in frequency table.
			for j := 0; j < xColsCat; j++ {
				km.FrequencyTable[int(km.Labels.At(i, 0))][j][rowCat.At(j, 0)]--
				km.FrequencyTable[int(newLabel)][j][rowCat.At(j, 0)]++

			}
			change = true

			numOfChanges++
			changed[int(newLabel)] = true
			changed[int(km.Labels.At(i, 0))] = true
			km.Labels.SetVec(i, newLabel)
		}

	}

	// Recompute cluster centers for all clusters with changes.
	for i, elem := range changed {
		if elem == true {
			// Find new values for clusters centers.
			km.findNewCenters(xColsCat, xNumCols, i, xNum)

		}
	}

	return totalCost, change, nil
}

func (km *KPrototypes) findNewCenters(xColsCat, xNumCols, i int, xNum *DenseMatrix) {
	newCentroid := make([]float64, xColsCat)
	for j := 0; j < xColsCat; j++ {
		val, empty := findHighestMapValue(km.FrequencyTable[i][j])
		if !empty {
			newCentroid[j] = val
		} else {
			newCentroid[j] = km.ClusterCentroidsCat.At(i, j)
		}
	}
	km.ClusterCentroidsCat.SetRow(i, newCentroid)

	vecSum := make([]*mat.VecDense, km.ClustersNumber)
	for a := 0; a < km.ClustersNumber; a++ {
		vecSum[a] = mat.NewVecDense(xNumCols, nil)
	}

	for a := 0; a < km.ClustersNumber; a++ {
		newCenter := make([]float64, xNumCols)
		for j := 0; j < km.LabelsCounter[a]; j++ {
			for k := 0; k < xNumCols; k++ {
				vecSum[a].SetVec(k, vecSum[a].At(k, 0)+xNum.At(int(km.MembershipNumTable[a][j]), k))
			}

		}
		for l := 0; l < xNumCols; l++ {
			newCenter[l] = vecSum[a].At(l, 0) / float64(km.LabelsCounter[a])
		}

		km.ClusterCentroidsNum.SetRow(a, newCenter)
	}
}

func (km *KPrototypes) near(index int, vectorCat, vectorNum *DenseVector) (float64, float64, error) {
	var newLabel, distance float64
	distance = math.MaxFloat64

	for i := 0; i < km.ClustersNumber; i++ {
		distCat, err := km.DistanceFunc(vectorCat, km.ClusterCentroidsCat.RowView(i).(*DenseVector))
		if err != nil {
			return -1, -1, fmt.Errorf("Cannot compute nearest cluster for vector %q: %v", index, err)
		}
		distNum, err := EuclideanDistance(vectorNum, km.ClusterCentroidsNum.RowView(i).(*DenseVector))
		if err != nil {
			return -1, -1, fmt.Errorf("Cannot compute nearest cluster for vector %q: %v", index, err)
		}
		dist := distCat + km.Gamma*distNum
		if dist < distance {
			distance = dist
			newLabel = float64(i)
		}
	}
	return newLabel, distance, nil
}

// Predict assign labels for the set of new vectors.
func (km *KPrototypes) Predict(X *DenseMatrix) (*DenseVector, error) {
	if km.IsFitted != true {
		return NewDenseVector(0, nil), errors.New("kmodes: cannot predict labels, model is not fitted yet")
	}
	xRows, xCols := X.Dims()
	labelsVec := NewDenseVector(xRows, nil)

	// Split data on categorical and numerical.
	xCat := NewDenseMatrix(xRows, len(km.CategoricalInd), nil)
	xNum := NewDenseMatrix(xRows, xCols-len(km.CategoricalInd), nil)
	var lastCat, lastNum int
	for i := 0; i < xCols; i++ {
		vec := make([]float64, xRows)
		vec = mat.Col(vec, i, X)

		if km.CategoricalInd[lastCat] == i {
			xCat.SetCol(lastCat, vec)
			lastCat++
			if lastCat >= len(km.CategoricalInd) {
				lastCat--
			}
		} else {
			xNum.SetCol(lastNum, vec)
			lastNum++
		}
	}

	// Normalize numerical values.
	xNum = normalizeNum(xNum)

	for i := 0; i < xRows; i++ {
		catVector := xCat.RowView(i).(*DenseVector)
		numVector := xNum.RowView(i).(*DenseVector)
		label, _, err := km.near(i, catVector, numVector)
		if err != nil {
			return NewDenseVector(0, nil), fmt.Errorf("kmodes Predict: %v", err)
		}
		labelsVec.SetVec(i, label)
	}

	return labelsVec, nil
}

// SaveModel saves computed ml model (KPrototypes struct) in file specified in
// configuration.
func (km *KPrototypes) SaveModel() error {
	file, err := os.Create(km.ModelPath)
	if err == nil {
		encoder := gob.NewEncoder(file)
		encoder.Encode(km)
	}
	file.Close()
	return err
}

// LoadModel loads model (KPrototypes struct) from file.
func (km *KPrototypes) LoadModel() error {
	file, err := os.Open(km.ModelPath)
	if err == nil {
		decoder := gob.NewDecoder(file)
		err = decoder.Decode(&km)
	}
	file.Close()
	SetWeights(km.WeightVectors[0])
	return err
}

func normalizeNum(X *DenseMatrix) *DenseMatrix {
	xRows, xCols := X.Dims()
	for i := 0; i < xCols; i++ {
		column := X.ColView(i).(*mat.VecDense).RawVector().Data
		max := maxVal(column)
		for j := 0; j < xRows; j++ {
			X.Set(j, i, X.At(j, i)/max)
		}
	}
	return X
}

func (km *KPrototypes) validateParameters() error {
	if km.InitializationFunc == nil {
		return errors.New("initializationFunction is nil")
	}
	if km.DistanceFunc == nil {
		return errors.New("distanceFunction is nil")
	}
	if km.ClustersNumber < 1 || km.MaxIterationNumber < 1 || km.RunsNumber < 1 {
		return errors.New("wrong initialization parameters (should be >1)")
	}
	return nil
}
