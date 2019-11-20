require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const plot = require('node-remote-plot');


const {features, labels, testFeatures, testLabels} = loadCSV('../data/cars.csv', {
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['passedemissions'],
    shuffle: true,
    splitTest: 50,
    converters: {
        passedemissions:(value) => {
            return value === 'TRUE' ? 1 : 0;
        }
    }
})

features_tensor = tf.tensor(features);
labels_tensor = tf.tensor(labels);
testFeatures_tensor =  tf.tensor(testFeatures);
testLabels_tensor =  tf.tensor(testLabels);

console.log(features_tensor);
console.log(labels_tensor);

const model = tf.sequential({
    layers: [
      tf.layers.dense({inputShape: [3], units: 16, activation: 'sigmoid'}),
      tf.layers.dense({units: 1, activation: 'sigmoid'}),
    ]
   });


model.summary();
console.log(model.compile({
    optimizer: 'sgd',
  loss: 'meanSquaredError',
  metrics: ['accuracy']
}));




  // Train for 5 epochs with batch size of 32.
  model.fit(features_tensor, labels_tensor, {
     epochs: 5,
     batchSize: 50,
   }).then(info => {
    console.log('Final accuracy', info.history.acc);
  });

//model.predict(tf.tensor([[150, 400 ,1.88]])).print();