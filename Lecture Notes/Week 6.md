# Image Recognition

In section 12 we move over to another problem this time involving image recognition. Here we go over how image data is represented and how to format it to go through our regression algorithm. The dataset that we are using is the mnist dataset which is a collection of 28 by 28 pixel hand written digits between 0 and 9 that we are going to classify.

Since this is a clasification type problem we had to first do a few changes to our algorithm such as encoding our labels into a onehot vector. There was also some computational bugs tha we had to deal with that results in NaNs. Once those issues were solved i was able to get a 90 percent accuracy on a limited trainig and test size set. There were a lot of optimiation problems in the code that we addressed in the section 13.

# Optimization

On my machine i wasn't at first able to compute the whole mnist dataset and I was still using all 32gb of my ram.
In this section we used the debbuger and a few well placed tf.tidys to optimize our code. At the end of the section I was able to run the whole dataset but while still using a considerable amount of memory. My final accuracy was about 85% on the whole set.

# TensorFlow.js: Layers API

In the video that we were tasked to watch, we were shown an example of setting up a neural network similar to the one that we created in class. It goes over creating a sequential model, adding layers, configuring those layers, setting up our loss function and compiling the model.
