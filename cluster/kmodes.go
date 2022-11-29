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

// DistanceFunction compute distance between two vectors.
type DistanceFunction func(a, b *DenseVector) (float64, error)

// InitializationFunction compute initial vales for cluster_centroids_.
type InitializationFunction func(X *DenseMatrix, clustersNumber int, distFunc DistanceFunction) (*DenseMatrix, error)

// KModes is a basic class for the k-modes algorithm, it contains all necessary
// information as alg. parameters, labels, centroids, ...
type KModes struct {
	DistanceFunc       DistanceFunction
	InitializationFunc InitializationFunction
	ClustersNumber     int
	RunsNumber         int
	MaxIterationNumber int
	WeightVectors      [][]float64
	FrequencyTable     [][]map[float64]float64 // frequency table - list of lists with dictionaries containing frequencies of values per cluster and attribute
	LabelsCounter      []int
	Labels             *DenseVector
	ClusterCentroids   *DenseMatrix
	IsFitted           bool
	ModelPath          string
}

// NewKModes implements constructor for the KModes struct.
func NewKModes(dist DistanceFunction, init InitializationFunction, clusters int, runs int, iters int, weights [][]float64, modelPath string) *KModes {
	rand.Seed(time.Now().UnixNano())
	return &KModes{
		DistanceFunc:       dist,
		InitializationFunc: init,
		ClustersNumber:     clusters,
		RunsNumber:         runs,
		MaxIterationNumber: iters,
		WeightVectors:      weights,
		ModelPath:          modelPath,
		Labels:             &DenseVector{VecDense: new(mat.VecDense)},
		ClusterCentroids:   &DenseMatrix{Dense: new(mat.Dense)},
	}
}

// FitModel main algorithm function which finds the best clusters centers for
// the given dataset X.
//func (km *KModes) FitModel(X *mat.Dense) error {
func (km *KModes) FitModel(X *DenseMatrix) error {
	err := km.validateParameters()
	if err != nil {
		return fmt.Errorf("kmodes: failed to fit the model: %v", err)
	}
	// Initialize weightVector
	SetWeights(km.WeightVectors[0])

	xRows, xCols := X.Dims()

	// Initialize clusters
	km.ClusterCentroids, err = km.InitializationFunc(X, km.ClustersNumber, km.DistanceFunc)
	if err != nil {
		return fmt.Errorf("kmodes: failed to fit the model: %v", err)
	}

	// Initialize labels vector
	km.Labels = NewDenseVector(xRows, nil)
	km.LabelsCounter = make([]int, km.ClustersNumber)

	// Create frequency table
	km.FrequencyTable = make([][]map[float64]float64, km.ClustersNumber)
	for i := range km.FrequencyTable {
		km.FrequencyTable[i] = make([]map[float64]float64, xCols)
		for j := range km.FrequencyTable[i] {
			km.FrequencyTable[i][j] = make(map[float64]float64)
		}
	}

	// Perform initial assignements to clusters - in order to fill in frequency
	// table.
	for i := 0; i < xRows; i++ {
		row := X.RowView(i)
		newLabel, _, err := km.near(i, &DenseVector{X.RowView(i).(*mat.VecDense)})
		km.LabelsCounter[int(newLabel)]++
		km.Labels.SetVec(i, newLabel)
		if err != nil {
			return fmt.Errorf("kmodes: initial labels assignement failure: %v", err)
		}
		for j := 0; j < xCols; j++ {
			km.FrequencyTable[int(newLabel)][j][row.At(j, 0)]++
		}

	}

	// Perform initial centers update - because iteration() starts with label
	// assignements.
	for i := 0; i < km.ClustersNumber; i++ {
		km.findNewCenters(i, xCols)
	}

	for i := 0; i < km.MaxIterationNumber; i++ {
		_, change, err := km.iteration(X)
		if err != nil {
			return fmt.Errorf("KMeans error at iteration %d: %v", i, err)
		}
		if !change {

			km.IsFitted = true
			return nil
		}
	}

	return nil
}

func (km *KModes) findNewCenters(i, xCols int) {

	newCentroid := make([]float64, xCols)
	for j := 0; j < xCols; j++ {
		val, empty := findHighestMapValue(km.FrequencyTable[i][j])
		if !empty {
			newCentroid[j] = val
		} else {
			newCentroid[j] = km.ClusterCentroids.At(i, j)
		}

	}
	km.ClusterCentroids.SetRow(i, newCentroid)
}

func (km *KModes) iteration(X *DenseMatrix) (float64, bool, error) {
	changed := make([]bool, km.ClustersNumber)
	var change bool
	var numOfChanges float64
	var totalCost float64

	// Find closest cluster for all data vectors - assign new labels.
	xRows, xCols := X.Dims()

	for i := 0; i < xRows; i++ {
		row := X.RowView(i)
		newLabel, cost, err := km.near(i, &DenseVector{X.RowView(i).(*mat.VecDense)})
		if err != nil {
			return totalCost, change, fmt.Errorf("iteration error: %v", err)
		}
		totalCost += cost

		if newLabel != km.Labels.At(i, 0) {
			km.LabelsCounter[int(newLabel)]++
			km.LabelsCounter[int(km.Labels.At(i, 0))]--

			// Make changes in frequency table.
			for j := 0; j < xCols; j++ {
				km.FrequencyTable[int(km.Labels.At(i, 0))][j][row.At(j, 0)]--
				km.FrequencyTable[int(newLabel)][j][row.At(j, 0)]++

			}
			change = true

			numOfChanges++
			changed[int(newLabel)] = true
			changed[int(km.Labels.At(i, 0))] = true
			km.Labels.SetVec(i, newLabel)
		}

	}

	// Check for empty clusters - if such cluster is found reassign the center
	// and return.
	for i := 0; i < km.ClustersNumber; i++ {
		if km.LabelsCounter[i] == 0 {
			vector := X.RawRowView(rand.Intn(xRows))
			km.ClusterCentroids.SetRow(i, vector)
			return totalCost, true, nil
		}
	}

	// Recompute cluster centers for all clusters with changes.
	for i, elem := range changed {
		if elem {
			//find new values for clusters centers
			km.findNewCenters(i, xCols)
		}
	}

	return totalCost, change, nil
}

func (km *KModes) near(index int, vector *DenseVector) (float64, float64, error) {
	var newLabel, distance float64
	distance = math.MaxFloat64
	for i := 0; i < km.ClustersNumber; i++ {
		dist, err := km.DistanceFunc(vector, &DenseVector{km.ClusterCentroids.RowView(i).(*mat.VecDense)})
		if err != nil {
			return -1, -1, fmt.Errorf("cannot compute nearest cluster for vector %q: %v", index, err)
		}
		if dist < distance {
			distance = dist
			newLabel = float64(i)
		}
	}
	return newLabel, distance, nil
}

func findHighestMapValue(m map[float64]float64) (float64, bool) {
	var key float64
	var highestValue float64

	// Do something different if map is empty because if its empty it returns
	// key=0 !!!
	if len(m) == 0 {
		return 0, true
	}

	for k, value := range m {
		if value > highestValue {
			highestValue = value
			key = k

		}
	}

	return key, false
}

// Predict assign labels for the set of new vectors.
func (km *KModes) Predict(X *DenseMatrix) (*DenseVector, error) {
	if !km.IsFitted {
		return &DenseVector{&mat.VecDense{}}, errors.New("kmodes: cannot predict labels, model is not fitted yet")
	}
	xRows, _ := X.Dims()
	labelsVec := NewDenseVector(xRows, nil)
	for i := 0; i < xRows; i++ {
		label, _, err := km.near(i, &DenseVector{X.RowView(i).(*mat.VecDense)})
		if err != nil {
			return &DenseVector{&mat.VecDense{}}, fmt.Errorf("kmodes Predict: %v", err)
		}
		labelsVec.SetVec(i, label)
	}
	return labelsVec, nil
}

// SaveModel saves computed ml model (KModes struct) in file specified in
// configuration.
func (km *KModes) SaveModel() error {
	file, err := os.Create(km.ModelPath)
	if err == nil {
		encoder := gob.NewEncoder(file)
		encoder.Encode(km)
	}
	file.Close()
	return err
}

// LoadModel loads model (KModes struct) from file.
func (km *KModes) LoadModel() error {
	file, err := os.Open(km.ModelPath)
	if err == nil {
		decoder := gob.NewDecoder(file)
		err = decoder.Decode(km)
	}
	file.Close()
	SetWeights(km.WeightVectors[0])
	return err
}

func (km *KModes) validateParameters() error {
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
