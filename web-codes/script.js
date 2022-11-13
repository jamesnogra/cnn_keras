// For getting the model
let model = undefined
async function loadModel() {
	model = await tf.loadLayersModel('keras-fruits-js/model.json')
}
loadModel()

// For preview of image after browse
function display(input) {
	if (input.files && input.files[0]) {
		const reader = new FileReader();
		reader.onload = function(e) {
			$('#input-image').attr('src', e.target.result);
		}
		reader.readAsDataURL(input.files[0]);
	}
}

$("#filePhoto").change(function() {
  readURL(this);
});

// When clicking the predict button
async function predict() {
	let input = document.getElementById('input-image')
	let step1 = tf.browser
		.fromPixels(input)
		.resizeNearestNeighbor([128, 128])
		.div(tf.scalar(255))
		.expandDims(0)
	const pred = model.predict(step1).arraySync()
	const prob = tf.softmax(pred).arraySync()[0]
	displayResult(prob)
}

// Display the result
function displayResult(prob) {
	console.log(prob)
}