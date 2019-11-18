# Natural Binary Classification

In this section we looked at a classification type problem. Instead of using linear regression like we did before since we had a continuous output, we have to use logistic regression.

logistic regression is very similar to linear regression. The only things that change are our cost function and the use of the Sigmoid (logistics) function.

At first the classification problem that we were given was binary in that it was only between two different options such as pass and fail. We were to output pass and fail of an emissions test depending on a cars given horsepower, engine displacement, and weight.

In order to get outputs between 0 and 1 we made use of the sigmoid function since our usual equation of `m*x+b` was not suitable for this.

later on we found that our mean squared function was actually not the appropriate cost function for classifications. The cost function that we ended up implementing was the cross entropy function.

# Multinomial Logistic Regression

In this section we took our Logistic regression model and used it to predict multiple outputs instead of just one. Th way it kind of works out is that we just do our same logistic regression but multiple at the same time. The outputs that we got from the model were marginal probabilities as apposed to conditional probabilities. The differences are stated below.

Marginal probability - considers one possible output case in isolation

* Sigmoid function used

Conditional probability - considers all possible output cases together

* SoftMax function is used

As the description of the differences says the in order to get a Conditional probability we had to use a function called SoftMax which takes into account all of the cases probabilities to create one probability between the cases that when summed up equals 1.

# Gradient descent, how neural networks learn
In 3Blue1Brown's video he goes over the how Gradient decent works. When a neural network model is being trained it at first puts out random values which are probably completely useless. In order to get useful outputs we need to train the network. In order to tell how wrong our network is we have to use a cost function. In the case of the video we used the mean squared error function. In order minimize this cost function we used gradient decent which tells us the direction we need to go that will decrease the cost function the fastest. Afterwards backpropagation is used to actually prorogate those changes through our network.
