require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinerRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let {features, labels, testFeatures, testLabels} = loadCSV('./cars.csv',{
   shuffle: true,
   splitTest: 50,
   dataColumns: ['horsepower', 'weight', 'displacement'],
   labelColumns: ['mpg']
});

const regression = new LinerRegression(features, labels,{
   
   learningRate : 0.1,
   iterations: 3,
   batchSize: 10,
});

//regression.features.print();

regression.train();
const r2 = regression.test(testFeatures,testLabels)
//console.log('MSE History', regression.mseHistory);

plot({
   x: regression.mseHistory.reverse(),
   xLabel: 'Iteration #',
   yLabel: 'Mean Squared Error'
});
console.log('R2 is', r2);
//console.log(regression.weights.arraySync());
//console.log('Updated M is:', regression.weights.arraySync()[0,1], 'Updated B is:', regression.weights.arraySync()[0,0]);
regression.predict([
   //['horsepower', 'weight', 'displacement']
   [120,2, 380],
   [135, 2.1, 420]
]).print()
