
const a = tf.tensor([1,2,3,4]);
const b = tf.tensor([5,6,7,8]);
const d = tf.tensor([9,3,4,5]);

const test_add = a.add(b);
test_add.print();

const test_sub = b.sub(a);
test_sub.print();

const test_mul = a.mul(b);
test_mul.print();

const test_div = b.div(a);
test_div.print();

const test_concat = a.reshape([4,1]).concat(b.reshape([4,1]),1);
test_concat.print();

const c = test_concat;
const e = c.concat(d.reshape([4,1]),1);
e.print();

const test_slice = e.slice([0,1],[-1,1]);
test_slice.print();

e.sum(1).print();

e.expandDims(2).print();

e.array().then(array => console.log(array));

