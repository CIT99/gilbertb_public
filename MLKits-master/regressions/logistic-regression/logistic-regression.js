const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression{
    constructor(features, labels, options){
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.costHistory = [];
        //this.bHistory = [];
        

        this.options = Object.assign(
            {learningRate: 0.1, iterations: 1000 , decisionBoundary: 0.5},
            options);
        
        this.weights = tf.zeros([this.features.shape[1], 1]);

    }

    gradientDescent(features,labels){
        const currentGuesses = features.matMul(this.weights).sigmoid();
        const differences = currentGuesses.sub(labels);

        const slopes = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0])
        
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate))
    }
    /*
    gradientDescent(){
        const currentGuessesForMPG = this.features.map(row =>{
            return this.m * row[0] + this.b;
        });

        const bSlope = _.sum(currentGuessesForMPG.map((guess,i) =>{
            return guess - this.labels[i][0];
        })) * 2 / this.features.length;

        const mSlope = _.sum(currentGuessesForMPG.map((guess,i) =>{
            return -1 * this.features[i][0] * (this.labels[i][0]-guess);
        })) * 2 / this.features.length;

        this.m = this.m - mSlope * this.options.learningRate;
        this.b = this.b - bSlope * this.options.learningRate;
    }
*/
    

    train(){
        const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);
        
        for (let i = 0; i < this.options.iterations; i++){
            for (let j = 0; j< batchQuantity;j++){
                const startIndex = j * this.options.batchSize;
                const {batchSize} = this.options;
                const featureSlice = this.features.slice([startIndex, 0],[batchSize, -1])

                const labelSlice = this.labels.slice([startIndex, 0],[batchSize, -1])
                this.gradientDescent(featureSlice,labelSlice);
            }

            this.recordCost();
            this.updatelearningRate();

        }
    }

    predict(observations){
        return this.processFeatures(observations)
            .matMul(this.weights)
            .sigmoid()
            .greater(this.options.decisionBoundary)
            .cast('float32');
    }

    test(testFeatures, testLabels){
        const predictions = this.predict(testFeatures);
        testLabels = tf.tensor(testLabels);

        const incorrect = predictions
            .sub(testLabels)
            .abs()
            .sum()
            .arraySync();
        //console.log(incorrect);
        return (predictions.shape[0] - incorrect) / predictions.shape[0];
    }
/*
    test(testFeatures, testLabels){
        testFeatures = this.processFeatures(testFeatures);
        testLabels = tf.tensor(testLabels);

        
        const predictions = testFeatures.matMul(this.weights);
        //predictions.print()

        const res = testLabels.sub(predictions)
            .pow(2)
            .sum()
            .arraySync()
        const tot = testLabels.sub(testLabels.mean())
            .pow(2)
            .sum()
            .arraySync()
        
        return 1 - res / tot;
    }
*/

    processFeatures(features){
        features = tf.tensor(features);
        
        if (this.mean && this.variance){
            features = features.sub(this.mean).div(this.variance.pow(0.5));
        }else{
            features = this.standardize(features)
        }

        features = tf.ones([features.shape[0],1]).concat(features, 1);

        return features;
    }

    standardize(features){
        const {mean, variance} = tf.moments(features, 0);

        this.mean = mean;
        this.variance = variance;

        return features.sub(mean).div(variance.pow(0.5));
    }
/*
    recordMSE(){
        const mse = this.features
            .matMul(this.weights)
            .sub(this.labels)
            .pow(2)
            .sum()
            .div(this.features.shape[0])
            .arraySync();
        this.mseHistory.unshift(mse);
    }
    */
    recordCost(){
        const guesses = this.features.matMul(this.weights).sigmoid();
        const termOne = this.labels
            .transpose()
            .matMul(guesses.log());
        const termTwo = this.labels
            .mul(-1)
            .add(1)
            .transpose()
            .matMul(
                guesses.mul(-1)
                    .add(1)
                    .log()
            );
            const cost = termOne.add(termTwo)
                .div(this.features.shape[0])
                .mul(-1)
                .arraySync()[0][0];

            this.costHistory.unshift(cost)
    }

    updatelearningRate(){
        if (this.costHistory.length < 2){
            return;
        }

        if (this.costHistory[0] > this.costHistory[1]){
            this.options.learningRate /= 2;
        }else{
            this.options.learningRate *= 1.05;
        }
    }
}

module.exports = LogisticRegression;