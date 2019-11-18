# Week 2 notes

## Section 3

In this section we cover the basics of the Tensorflow library.

1. The difference between Shape and Dimension
2. Elementwise operations
3. Broadcasting operations
4. Displaying and accessing tensor data
    * For instance slicing data with the slice method
    * syntax: `tf.data.slice([index_R, index_C],[size_R, size_C])`
5. Modifying tensors by concatenation and summing elements

## Section 4

In this section we implement our KNN algorithm again but using tensorflow

In this section we are given a regression type problem. If we are given the latitude and longitude of a house and the square feet can we predict the price of the house.

I tried to implement the knn algorithm on my machine but I'm having problems with syntax or libraries.

In the video lectures what we were supposed to do was similar to what we did in the plinko app. First we were supposed to load our CSV file, separate the data into a testSet and trainingSet, and instead of normalizing the data we standardized it.

The KNN function looked like this:

```javascript

function knn(features, labels, predictionPoint, k){
    const {mean, variance} = tf.moments(features, 0);
    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));
    return (
        features
            .sub(mean)
            .div(variance.pow(0.5))
            .sub(scaledPrediction)
            .pow(2)
            .sum(1)
            .pow(0.5)
            .expandDims(1)
            .concat(labels,1)
            .unstack()
            .sort((a,b)=> a.get(0)> b.get(0)? 1 : -1)
            .slice(0,k)
            .reduce((acc,pair)=> acc + pair.get(1),0)/k
    );
    
}
```
