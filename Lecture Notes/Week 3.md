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
  