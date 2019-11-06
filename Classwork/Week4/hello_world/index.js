const tf = require('@tensorflow/tfjs')

const model = tf.sequential()

model.add(tf.layers.dense({units:1, inputShape: [1]}))

//meanSquaredError - this is our loss function
//sgd - Stochastic Gradient Descent -> our optimizer for training

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'})

//tensor1d - creates one dim tensor

const xs = tf.tensor1d([-1.0, 0.0, 1.0, 2.0, 3.0,  4.0])
const ys = tf.tensor1d([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0])

model.fit(xs,ys, {epochs: 50}).then(()=>{
    model.predict(tf.tensor1d([11])).print()
})

