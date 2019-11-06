# Getting Started with Gradient Decent

## Linear Regression

* Fast We only need to train it one time
* uses methods that will be important in more complicated ML
* y = mx + b

In order to better approximate our guesses we use a loss function called Mean Squared Error

## Gradient Descent

1. pick value for b
2. calculate the slope of MSE with b
3. if slope is small then done
4. multiply slope by learning rate
5. subtract result from b and repeat

## class structure for linearRegression

* `gradientDescent()`
  * Runs one iteration GD and updates 'm' and 'b'
* `train()`
  * Run GD until we get good values for 'm' and 'b'
* `test()`
  * Use a 'test' data set to evaluate the accuracy of our calculated 'm' and 'b'
* `predict()`
  * Make a prediction using our calculated 'm' and 'b'
  
## Initial gradient Descent

  In our first iteration of the gradient decent function we implemented our Mean Squared function separately for our weights and our bias denoted as 'M' and 'B' respectively. Our function is located below:

  ```javascript
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
  ```

## Gradient Descent with Tensorflow

We began to implement the gradient decent function using tensorflow by beginning to grasp the basic concepts of matrix multiplication.
