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
	Labels              *mat.VecDense
	ClusterCentroids    *mat.Dense
	ClusterCentroidsCat *mat.Dense
	ClusterCentroidsNum *mat.Dense
	Gamma               float64
	IsFitted            bool
	ModelPath           string
}

// NewKPrototypes implements constructor for the KPrototypes struct.
func NewKPrototypes(dist DistanceFunction, init InitializationFunction, categorical []int, clusters int, runs int, iters int, weights [][]float64, g float64, modelPath string) *KPrototypes {
	rand.Seed(time.Now().UnixNano())
	return &KPrototypes{DistanceFunc: dist, InitializationFunc: init, ClustersNumber: clusters, CategoricalInd: categorical, RunsNumber: runs, MaxIterationNumber: iters, Gamma: g, WeightVectors: weights, ModelPath: modelPath}
}

// FitModel main algorithm function which finds the best clusters centers for
// the given dataset X.
func (km *KPrototypes) FitModel(X *mat.Dense) error {
	if km.InitializationFunc == nil {
		return errors.New("kprototypes: failed to fit the model: InitializationFunction is nil")
	}
	if km.DistanceFunc == nil {
		return errors.New("kprototypes: failed to fit the model: DistanceFunction is nil")
	}
	if km.ClustersNumber < 1 || km.MaxIterationNumber < 1 || km.RunsNumber < 1 {
		return errors.New("kprototypes: failed to fit the model: wrong initialization parameters (should be >1)")
	}
	xRows, xCols := X.Dims()

	// Partition data on two sets - one with categorical, other with numerical
	// data.
	xCat := mat.NewDense(xRows, len(km.CategoricalInd), nil)
	xNum := mat.NewDense(xRows, xCols-len(km.CategoricalInd), nil)
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

	_, xCatCols := xCat.Dims()
	_, xNumCols := xNum.Dims()

	// Normalize numerical values.
	xNum = normalizeNum(xNum)

	// Initialize weightVector.
	SetWeights(km.WeightVectors[0])

	// Initialize clusters for categorical data.
	var err error
	km.ClusterCentroidsCat, err = km.InitializationFunc(xCat, km.ClustersNumber, km.DistanceFunc)
	if err != nil {
		return fmt.Errorf("kmodes: failed to fit the model: %v", err)
	}

	// Initialize clusters for numerical data.
	km.ClusterCentroidsNum, err = InitNum(xNum, km.ClustersNumber, km.DistanceFunc)

	// Initialize labels vector
	km.Labels = mat.NewVecDense(xRows, nil)
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
		rowCat := xCat.RowView(i).(*mat.VecDense)
		rowNum := xNum.RowView(i).(*mat.VecDense)
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
		newCatCentroid := make([]float64, xCatCols)

		for j := 0; j < xCatCols; j++ {
			val, empty := findHighestMapValue(km.FrequencyTable[i][j])
			if !empty {
				newCatCentroid[j] = val
			} else {
				newCatCentroid[j] = km.ClusterCentroidsCat.At(i, j)
			}
		}
		km.ClusterCentroidsCat.SetRow(i, newCatCentroid)
	}

	//newNumCentroid := make([]float64, xNumCols)
	vecSum := make([]*mat.VecDense, km.ClustersNumber)
	for i := 0; i < km.ClustersNumber; i++ {
		vecSum[i] = mat.NewVecDense(xNumCols, nil)
	}

	for i := 0; i < km.ClustersNumber; i++ {
		newCenter := make([]float64, xNumCols)
		for j := 0; j < km.LabelsCounter[i]; j++ {
			for k := 0; k < xNumCols; k++ {
				vecSum[i].SetVec(k, vecSum[i].At(k, 0)+xNum.At(int(km.MembershipNumTable[i][j]), k))
			}

		}
		for l := 0; l < xNumCols; l++ {

			newCenter[l] = vecSum[i].At(l, 0) / float64(km.LabelsCounter[i])
		}
		km.ClusterCentroidsNum.SetRow(i, newCenter)
	}

	//var lastCost float64
	//lastCost = math.MaxFloat64

	for i := 0; i < km.MaxIterationNumber; i++ {
		_, change, err := km.iteration(xNum, xCat)
		if err != nil {
			return fmt.Errorf("KMeans error at iteration %d: %v", i, err)
		}
		//lastCost = cost
		//if cost > lastCost || change == false {
		if change == false {
			km.IsFitted = true
			return nil
		}
	}

	return nil
}

func (km *KPrototypes) iteration(xNum, xCat *mat.Dense) (float64, bool, error) {
	changed := make([]bool, km.ClustersNumber)
	var change bool
	var numOfChanges float64
	var totalCost float64

	for i := 0; i < km.ClustersNumber; i++ {
		km.MembershipNumTable[i] = nil
	}

	// Find closest cluster for all data vectors - assign new labels.
	xRowsNum, xNumCols := xNum.Dims()
	//xRowsCat, xColsCat := xCat.Dims()
	_, xColsCat := xCat.Dims()

	for i := 0; i < xRowsNum; i++ {
		rowCat := xCat.RowView(i).(*mat.VecDense)
		rowNum := xNum.RowView(i).(*mat.VecDense)
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

	/*//check for empty clusters - if such cluster is found reassign the center and return
	for i := 0; i < km.ClustersNumber; i++ {
		if km.LabelsCounter[i] == 0 {
			fmt.Println("oh no, there is an empty cluster! ", km.ClusterCentroidsCat.RowView(i))
			num := rand.Intn(xRowsCat)
			vectorCat := xCat.RawRowView(num)
			vectorNum := xNum.RawRowView(num)
			fmt.Println("New vectors are: ", vectorCat, vectorNum)
			km.ClusterCentroidsCat.SetRow(i, vectorCat)
			km.ClusterCentroidsNum.SetRow(i, vectorNum)
			return totalCost, true, nil
		}
	}*/

	// Recompute cluster centers for all clusters with changes.
	for i, elem := range changed {
		if elem == true {
			// Find new values for clusters centers.
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
	}

	return totalCost, change, nil
}

func (km *KPrototypes) near(index int, vectorCat, vectorNum *mat.VecDense) (float64, float64, error) {
	var newLabel, distance float64
	distance = math.MaxFloat64

	for i := 0; i < km.ClustersNumber; i++ {
		distCat, err := km.DistanceFunc(vectorCat, km.ClusterCentroidsCat.RowView(i).(*mat.VecDense))
		if err != nil {
			return -1, -1, fmt.Errorf("Cannot compute nearest cluster for vector %q: %v", index, err)
		}
		distNum, err := EuclideanDistance(vectorNum, km.ClusterCentroidsNum.RowView(i).(*mat.VecDense))
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
func (km *KPrototypes) Predict(X *mat.Dense) (*mat.VecDense, error) {
	if km.IsFitted != true {
		return mat.NewVecDense(0, nil), errors.New("kmodes: cannot predict labels, model is not fitted yet")
	}
	xRows, xCols := X.Dims()
	labelsVec := mat.NewVecDense(xRows, nil)

	// Split data on categorical and numerical.
	xCat := mat.NewDense(xRows, len(km.CategoricalInd), nil)
	xNum := mat.NewDense(xRows, xCols-len(km.CategoricalInd), nil)
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
		catVector := xCat.RowView(i).(*mat.VecDense)
		numVector := xNum.RowView(i).(*mat.VecDense)
		label, _, err := km.near(i, catVector, numVector)
		if err != nil {
			return mat.NewVecDense(0, nil), fmt.Errorf("kmodes Predict: %v", err)
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

// LoadModel loads model (KPrototypes struct) from file, it is invoked while
// 'training mode' is not used.
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

func normalizeNum(X *mat.Dense) *mat.Dense {
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
