package cluster

import (
	"errors"
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

var (
	weightVector *mat.VecDense
)

//HammingDistance is a basic dissimilarity function for the kmodes algorithm
func HammingDistance(a, b *mat.VecDense) (float64, error) {
	if a.Len() != b.Len() {
		return -1, errors.New("hamming distance: vectors lengths do not match")
	}
	var distance float64
	for i := 0; i < a.Len(); i++ {
		if a.At(i, 0) != b.At(i, 0) {
			distance++
		}
	}
	return distance, nil
}

//WeightedHammingDistance dissimilarity function is based on hamming distance but it adds improttance to attributes
func WeightedHammingDistance(a, b *mat.VecDense) (float64, error) {
	if a.Len() != b.Len() {
		return -1, errors.New("hamming distance: vectors lengths do not match")
	}
	if a.Len() != weightVector.Len() {
		return -1, fmt.Errorf("weighted hamming distance: wrong weight vector length: %d", weightVector.Len())
	}

	var distance float64
	for i := 0; i < a.Len(); i++ {
		if a.At(i, 0) != b.At(i, 0) {
			distance += 1 * weightVector.At(i, 0)
		}
	}
	return distance, nil
}

//EuclideanDistance computes eucdlidean distance between two vectors
func EuclideanDistance(a, b *mat.VecDense) (float64, error) {
	if a.Len() != b.Len() {
		return -1, errors.New("euclidean distance: vectors lengths do not match")
	}
	var distance float64
	for i := 0; i < a.Len(); i++ {
		diff := (a.At(i, 0) - b.At(i, 0))
		distance += diff * diff
	}

	return math.Sqrt(distance), nil

}

// SetWeights sets the weight vector used in WeightedHammingDistance function
func SetWeights(newWeights []float64) {
	weightVector = mat.NewVecDense(len(newWeights), newWeights)
}

// ComputeWeights derives weights based on the frequency of attribute values (more different values means lower weight)
func ComputeWeights(X *mat.Dense, imp float64) []float64 {
	xRows, xCols := X.Dims()

	weights := make([]float64, xCols)

	for i := 0; i < xCols; i++ {
		column := X.ColView(i)
		frequencies := make(map[float64]float64)
		for j := 0; j < xRows; j++ {
			frequencies[column.At(j, 0)] = frequencies[column.At(j, 0)] + 1
		}

		if w := 1 / float64(len(frequencies)); w == 1 {
			weights[i] = 0
		} else {
			weights[i] = w
		}

	}
	m := maxVal(weights)
	mult := imp / m
	for i := range weights {
		weights[i] *= mult
	}

	return weights
}

func maxVal(table []float64) float64 {
	max := 0.0
	for _, e := range table {
		if e > max {
			max = e
		}
	}
	return max
}
