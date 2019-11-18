# Section Two Notes

For the second section of the course we spent our time analyzing the plinko game. The question we were trying to answer was given a drop position what was the probability of a ball falling into 1 of 10 buckets. these balls also had some variations such as size and bounciness.

Since our output for this problem was discrete, the number of the bucket that the ball fell into, we chose an algorithm that was best suited for classification problems, KNN or K-Nearest Neighbor.

The premise of the algorithm is that given a set of data which inputs are similar and the outputs are similar, if we compare another piece of data with similar inputs we can assume that our output will be similar. As the video lecture put it "birds of a feather flock together".

To begin with we were tasked with creating a simple KNN implementation which looked like this:

```javascript
const outputs = [
    [10 ,.5,16,1],
    [200,.5,16,4],
    [350,.5,16,4],
    [600,.5,16,6],
];

const predictionPoint = 300;
const k = 3;

function distance(point){
    return Math.abs(point - predictionPoint)
}

_.chain(outputs)
    .map(row => [distance(row[0]), row[3]])
    .slice(0,k)
    .countBy(row => row[1])
    .toPairs()
    .sortBy(row => row[1])
    .last()
    .first()
    .parseInt()
    .value()
```

This implementation allowed us to take our set of inputs and from out hard coded prediction point predict the probability of where the balls were going to fall. From our first few tests we saw that our predictions weren't that good so we implemented a way to better test our result by creating test data sets and training data sets.The code to split our dataset looked like this:

```javascript
function splitDataset(data, testCount){
  const shuffled = _.shuffle(data);
  const testSet = _.slice(shuffled, 0, testCount);
  const  trainingSet = _.slice(shuffled,testCount);

  return [testSet, trainingSet];
}
```

 From there we went on to improve our KNN algorithm to be more generalized and work on datasets with multiple features. We once again ran into a problem with our data where the difference in variation between our features was possibly causing our results to be skewed. From there we created a feature normalization function to help format our data sets. the normalization function called minMax looked like this:

```javascript

function minMax(data, featureCount){
  const clonedData = _.cloneDeep(data);
  for(let i = 0; i < featureCount; i++){
    const column = clonedData.map(row => row[i]);
    const min = _.min(column);
    const max = _.max(column);

    for(let j = 0; j < clonedData.length; j++){
      clonedData[j][i] = (clonedData[j][i] - min)/(max-min);
    }
  }
  return clonedData;
}
```

it took our dataset and the number of features as inputs then returned a normalized dataset ready to be processed further.

After adding the new features we saw that our prediction was still pretty low. From our test we could actually see that our accuracy went down from adding all the features to our algorithm. The ball bounciness was affecting the bucket that the ball fell into but it add a lot of randomness causing our prediction to drop. Once we compared the effects of using each of the lables by themselves to predict the output we decided to remove ball bounciness as it only decreased accuracy.

## Main take away

1. Features vs Labels
2. Test dataset vs Training dataset
3. Feature Normalization
4. Feature Selection
5. Common data structures
