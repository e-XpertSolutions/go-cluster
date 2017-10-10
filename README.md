# go-cluster

[![GoDoc](https://godoc.org/github.com/e-XpertSolutions/go-cluster/cluster?status.png)](http://godoc.org/github.com/e-XpertSolutions/go-cluster/cluster)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-yellow.svg?style=flat)](https://github.com/e-XpertSolutions/go-cluster/blob/master/LICENSE)
[![GoReport](https://goreportcard.com/badge/github.com/e-XpertSolutions/go-cluster)](https://goreportcard.com/report/github.com/e-XpertSolutions/go-cluster)
[![Travis](https://travis-ci.org/e-XpertSolutions/go-cluster.svg?branch=master)](https://travis-ci.org/e-XpertSolutions/go-cluster)

GO implementation of clustering algorithms: k-modes and k-prototypes.

K-modes algorithm is very similar to well-known clustering algorithm k-means. The difference is how the distance is computed. In k-means Euclidean distance between two vectors is most commonly used. While it works well for numerical, continuous data it is not suitable to use it with categorical data as it is impossible to compute the distance between values like ‘Europe’ and ‘Africa’. This is why in k-modes, the Hamming distance between vectors is used - it shows how many elements of two vectors is different. It is a good alternative for one-hot encoding while dealing with large number of categories for one feature. K-prototypes is used to cluster mixed data (both categorical and numerical).

Implementation of algorithms is based on papers: [HUANG97](#references), [HUANG98](#references), [CAO09](#references) and partially inspired by python implementation of same algorithms: [KMODES](#references).

## Installation

```
go get -u gopkg.in/e-XpertSolutions/go-cluster.v1
```

## Usage

This is basic configuration and usage of KModes and KPrototypes algorithms. For more information please refer to the documentation.

```go
package main

import (
    "fmt"
    "github.com/e-XpertSolutions/go-cluster/cluster"
)

func main() {

    //input categorical data first must be dictionary-encoded to numbers - for example for values
    //"blue", "red", "green" it can be 1,2,3

    data := cluster.NewDenseMatrix(lineNumber, columnNumber, rawData)
    newData := cluster.NewDenseMatrix(newLineNumber, newColumnNumber, newRawData)


    //input parameters for the algorithm

    //distance and initialization functions may be chosen from the package or one may use 
    //custom functions with proper arguments
    distanceFunction := cluster.WeightedHammingDistance
    initializationFunction := cluster.InitCao

    //number of clusters and maximum number of iterations 
    clustersNumber := 5
    maxIteration := 20

    //weight vector - used to set importance of the features, bigger number means greater 
    //contribution to the cost function
    //vector must be of the same length as the number of features in dataset
    //it is not compulsory, if 'nil' then all features are treated equally (weight = 1)  
    weights := []float64{1,1,2}
    wvec := [][]float64{weights}

    //path to file where model will be saved or loaded from using LoadModel(), SaveModel()
    //if no need to load or save the model, can be set to empty string
    path = "km.txt"

    //KModes algorithm
    //initialization
    km := cluster.NewKModes(distanceFunction, initializationFunction, clustersNumber, 1, 
    maxIteration, wvec, "km.txt")


    //training
    //after training it is possible to access clusters centers vectors and computed labels
    //using km.ClusterCentroids and km.Labels
    err := km.FitModel(data)
    if err != nil {
        fmt.Println(err)
    }

    //predicting labels for new data
    newLabels, err := km.Predict(newData)
    if err != nil {
        fmt.Println(err)
    }


    //KPrototypes algorithm
    //it needs two more parameters than k-modes:
    //categorical - vector with numbers indicating columns with categorical features
    //gamma - float number, importance of cost contribution for numerical values
    categorical := []int{1} // means that only column number one contains categorical data
    gamma := 0.2 //cost from distance function for numerical data will be multiplied by 0.2

    //initialization
    kp := cluster.NewKPrototypes(distanceFunction, initializationFunction, categorical, 
    clustersNumber, 1, maxIteration, wvec, gamma, "km.txt")

    //training
    err := kp.FitModel(data)
    if err != nil {
        fmt.Println(err)
    }

    //predicting labels for new data
    newLabelsP, err := kp.Predict(newData)
    if err != nil {
        fmt.Println(err)
    }
}

```


## Contributing

Contributions are greatly appreciated. The project follows the typical
[GitHub pull request model](https://help.github.com/articles/using-pull-requests/)
for contribution.


## License

The sources are release under a BSD 3-Clause License. The full terms of that
license can be found in `LICENSE` file of this repository.

## References
[HUANG97]: Huang, Z.: Clustering large data sets with mixed numeric and
   categorical values, Proceedings of the First Pacific Asia Knowledge
   Discovery and Data Mining Conference, Singapore, pp. 21-34, 1997.

[HUANG98] Huang, Z.: Extensions to the k-modes algorithm for clustering
   large data sets with categorical values, Data Mining and Knowledge
   Discovery 2(3), pp. 283-304, 1998.

[CAO09] Cao, F., Liang, J, Bai, L.: A new initialization method for
   categorical data clustering, Expert Systems with Applications 36(7),
   pp. 10223-10228., 2009.

[KMODES] Python implementation of k-modes: https://github.com/nicodv/kmodes