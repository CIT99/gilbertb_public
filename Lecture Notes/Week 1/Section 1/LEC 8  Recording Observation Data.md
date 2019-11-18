# Recording Observation Data

for this video we were tasked with editing our `onScoreUpdate()` to look like the code below.

``` javascript
const outputs = [];

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
  console.log(outputs);
}
```
