async function run(){
    const houseData = await getData()
    //console.log(JSON.stringify(houseData));
    //console.log(houseData)
    
    const values = houseData.map(house =>({
        x: house.rooms,
        y: house.price,
    }))

    //console.log(values)
    tfvis.render.scatterplot(
        {name:'No of rooms price'},
        {values : values},
        {
            xlabel: 'N',
            ylabel: 'Price',
            height: 300
        });

    const model = createModel();
    tfvis.show.modelSummary({name:'Model Summary'}, model)
    const tensorData = convertToTensor(houseData);
    const {inputs, labels} = tensorData;

    await trainModel(model, inputs,labels)

}

function createModel(){
    const model = tf.sequential({
        layers: [
          tf.layers.dense({inputShape: [1], useBias: true, units: 1}),
          tf.layers.dense({units: 1, useBias: true}),
        ]
    });

    return model;
}

function convertToTensor(data){
    return tf.tidy(()=>{
        tf.util.shuffle(data)
        const inputs = data.map(d => d.rooms)
        const labels = data.map(d => d.price)
        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
        const labelTensor = tf.tensor2d(labels, [inputs.length, 1])

        const inputMax = inputTensor.max()
        const inputMin = inputTensor.min()
        const labelMax = labelTensor.max()
        const labelMin = labelTensor.min()

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))

        return{
            inputs: normalizedInputs,
            labels: normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }
    })
}

/*
async function getData(){
    const response = await fetch('https://raw.githubusercontent.com/meetnandu05/ml1/master/house.json');
    const rjson = await response.json();
    //console.log(JSON.stringify(rjson));
    const data = rjson.map(x =>({
        price :x.price,
        rooms :x.rooms
        })).filter(x =>( (x.price !== null) && (x.rooms !== null)))
    return data
}
*/

async function getData() {
    const houseDataReq = await fetch('https://raw.githubusercontent.com/meetnandu05/ml1/master/house.json')
    const houseData = await houseDataReq.json()
    const cleaned = houseData.map(house =>({
        price: house.Price,
        rooms: house.AvgAreaNumberofRooms
    })).filter(house => (house.price != null && house.rooms != null))
    return cleaned 
}

/*
async function trainModel(inputs, labels, model){
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse']
    });
    const batchSize = 25
    const epochs = 10

    return await model.fit(inputs, labels, {
        epochs,
        batchSize,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            {name: 'Training'},
            ['loss', 'mse'],
            {height: 300, callback: ['onEpochEnd']}
        )

     });
}
*/

async function trainModel(model, inputs, labels) {
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse']

    })
    const batchSize = 25
    const epochs = 20

    return await model.fit(inputs, labels, {
        batchSize,
        epochs, 
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            {name: 'Training Performance'},
            ['loss', 'mse'],
            {height: 300, callbacks: ['onEpochEnd']}
        )

    })
}

run();