package cluster

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"gonum.org/v1/gonum/mat"
)

// KV is a structure that holds key-value pairs of type float64.
type KV struct {
	Key   float64
	Value float64
}

// InitHuang implements initialization of cluster centroids based on the
// frequency of attributes as defined in paper written by Z.Huang in 1998.
func InitHuang(X *DenseMatrix, clustersNumber int, distFunc DistanceFunction) (*DenseMatrix, error) {
	_, xCols := X.Dims()
	centroids := NewDenseMatrix(clustersNumber, xCols, nil)

	freqTable := CreateFrequencyTable(X)
	for j := 0; j < clustersNumber; j++ {
		for i := 0; i < xCols; i++ {
			if len(freqTable[i]) > j {
				centroids.Set(j, i, freqTable[i][j].Key)
			} else {
				// Change to setting to randomly chosen value instead of first
				// one.
				centroids.Set(j, i, freqTable[i][0].Key)
			}
		}
	}

	return centroids, nil
}

// InitCao implements initialization of cluster centroids based on the frequency
// and density of attributes as defined in
//    "A new initialization method for categorical data clustering" by F.Cao(2009)
func InitCao(X *DenseMatrix, clustersNumber int, distFunc DistanceFunction) (*DenseMatrix, error) {
	xRows, xCols := X.Dims()
	centroids := NewDenseMatrix(clustersNumber, xCols, nil)

	// Compute density table and, int the same time find index of vector with
	// the highest density.
	highestDensityIndex := 0
	maxDensity := 0.0
	densityTable := make([]float64, xRows)
	for i := 0; i < xCols; i++ {
		freq := make(map[float64]int)
		for j := 0; j < xRows; j++ {
			freq[X.At(j, i)]++
		}
		for j := 0; j < xRows; j++ {
			densityTable[j] += float64(freq[X.At(j, i)]) / float64(xCols)
		}
	}
	for k := 0; k < xRows; k++ {
		densityTable[k] = densityTable[k] / float64(xRows)
		if densityTable[k] > maxDensity {
			maxDensity = densityTable[k]
			highestDensityIndex = k
		}
	}

	// Choose first cluster - vector with maximum density.
	centroids.SetRow(0, X.RawRowView(highestDensityIndex))
	fmt.Println(0, X.RawRowView(highestDensityIndex))

	// Find the rest of clusters centers.
	for i := 1; i < clustersNumber; i++ {
		dd := make([][]float64, i)
		for z := 0; z < i; z++ {
			dd[z] = make([]float64, xRows)
		}
		for j := 0; j < i; j++ {
			for k := 0; k < xRows; k++ {
				dist, err := distFunc(&DenseVector{X.RowView(k).(*mat.VecDense)}, &DenseVector{centroids.RowView(j).(*mat.VecDense)})
				if err != nil {
					return NewDenseMatrix(0, 0, nil), fmt.Errorf("cao init: cannot compute cluster: %v ", err)
				}
				dd[j][k] = densityTable[k] * dist
			}
		}

		indexMax := findIndexCao(xRows, i, dd)

		fmt.Println(i, X.RawRowView(indexMax))
		centroids.SetRow(i, X.RawRowView(indexMax))
	}

	return centroids, nil
}

func findIndexCao(xRows, i int, dd [][]float64) int {
	// Find minimum value for each column.
	minValuesTable := make([]float64, xRows)
	for j := 0; j < xRows; j++ {
		minValuesTable[j] = math.MaxFloat64
	}

	for j := 0; j < i; j++ {
		for k := 0; k < xRows; k++ {
			if dd[j][k] < minValuesTable[k] {
				minValuesTable[k] = dd[j][k]
			}
		}
	}

	// Find max value and its index among minValuesTable.
	maxVal := 0.0
	indexMax := 0
	for j := 0; j < xRows; j++ {
		if minValuesTable[j] > maxVal {
			maxVal = minValuesTable[j]
			indexMax = j
		}
	}

	return indexMax
}

// InitRandom randomly initializes cluster centers - vectors chosen from X table.
func InitRandom(X *DenseMatrix, clustersNumber int, distFunc DistanceFunction) (*DenseMatrix, error) {
	xRows, xCols := X.Dims()
	centroids := NewDenseMatrix(clustersNumber, xCols, nil)
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < clustersNumber; i++ {
		centroids.SetRow(i, X.RawRowView(rand.Intn(xRows)))
	}
	return centroids, nil
}

// InitNum initializes cluster centers for numerical data - random
// initialization.
func InitNum(X *DenseMatrix, clustersNumber int, distFunc DistanceFunction) (*DenseMatrix, error) {
	xRows, xCols := X.Dims()
	centroids := NewDenseMatrix(clustersNumber, xCols, nil)
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < clustersNumber; i++ {
		center := X.RawRowView(rand.Intn(xRows - 1))
		centroids.SetRow(i, center)
	}
	return centroids, nil
}

// CreateFrequencyTable creates frequency table for attributes in given matrix,
// it returns attributes in frequency descending order.
func CreateFrequencyTable(X *DenseMatrix) [][]KV {
	xRows, xCols := X.Dims()
	frequencyTable := make([][]KV, xCols)
	for i := 0; i < xCols; i++ {
		column := X.ColView(i)
		frequencies := make(map[float64]float64)
		for j := 0; j < xRows; j++ {
			frequencies[column.At(j, 0)] = frequencies[column.At(j, 0)] + 1
		}
		for k, v := range frequencies {
			frequencyTable[i] = append(frequencyTable[i], KV{k, v})
		}
		sort.Slice(frequencyTable[i], func(a, b int) bool {
			return frequencyTable[i][a].Value > frequencyTable[i][b].Value
		})
	}
	return frequencyTable
}
