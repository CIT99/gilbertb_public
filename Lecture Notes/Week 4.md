# Increasing Performance with Vectorized Solutions

In section 7 we first began by re implementing our gradient decent function using only tesnorflow operations. We implemented the MSE formula for both the slope and our bias.

Once our function was implemented we tested our program and found that our results were not that great. We then went on to implement a function to assess the accuracy of our results.

## Coefficients of Determination

In order to assess the accuracy of our model we used the formula: `R^2 = 1 - (SS_res/SS_tot)`

Once we implemented this function we saw that our accuracy was still pretty low so we implemented a standardization function to help with processing our data.

## Multivariate Regression

Our accuracy was still quite terrible at this point so we decided to implement the ability for our model to take into account multiple labels and data.

## Learning Rate Optimization

We found that to get a some what acceptable accuracy from our model we had to modify our learning rate manually until we found a sweet spot. This method would be inefficient if we had to work with datasets that needed extend periods of time t process so we implemented a function that automatically change our learning rate depending on our data.

## Plotting Learning Rate and MSE

In section 8 we briefly covered the topic of plotting data in JavaScript. We used a `plot()` function to create plot of our changing MSE values to help us visualize the performance of our model as each iteration was ran.

## Batch and Stochastic Gradient Decent

To further improve the performance of our model we implemented batch processing of our data. With batch processing we would update our M and B MSE values after each batch instead of after processing the whole dataset. This lead to our model reaching its optimal values. Once we implemented processing all we had to do to implement Stochastic Gradient Decent was was change our batch size to one which would change our weights for every element of our dataset.


## Video Notes 

Neural networks are a type of machine learning algorithm that is based on the human brain structure.