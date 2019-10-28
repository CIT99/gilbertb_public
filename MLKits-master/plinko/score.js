
const outputs = [];
//const predictionPoint = 300;
const k = 3;

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel]);

}

function runAnalysis() {
  // Write code here to analyze stuff

  let testSetSize = 10;
  const [testSet, trainingSet] = splitDataset(outputs,testSetSize);
  
  /*let numCorrect = 0;
  for(let i=0; i < testSet.length; i++){
    const bucket = knn(trainingSet,testSet[i][0]);
    //console.log(bucket,testSet[i][3]);
    if(bucket === testSet[i][3]){
      numCorrect++;
    }
  }
  console.log('Accuracy: ', numCorrect/testSetSize);
  */

  const accuracy = _.chain(testSet)
    .filter(testPoint => knn(trainingSet,testPoint[0]) === testPoint[3])
    .size()
    .divide(testSetSize)
    .value()

  console.log('Accuracy is', accuracy);

}

function knn (data, testPoint){
  return _.chain(data)
    .map(row => [distance(row[0],testPoint), row[3]])
    .slice(0,k)
    .countBy(row => row[1])
    .toPairs()
    .sortBy(row => row[1])
    .last()
    .first()
    .parseInt()
    .value();
}

function distance(piont,testPoint){
  return Math.abs(piont - testPoint)
}

function splitDataset(data, testCount){
  const shuffled = _.shuffle(data);
  const testSet = _.slice(shuffled, 0, testCount);
  const  trainingSet = _.slice(shuffled,testCount);

  return [testSet, trainingSet];
}
