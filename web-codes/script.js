// Maximum number of result of classes to show
const MAX_CLASS_TO_DISPLAY = 5

// Wait for the elements to load
$(document).ready(function() {
	$('#file-input-container').hide()
	loadModel()
})

// For getting the model
let model
async function loadModel() {
	model = await tf.loadLayersModel('keras-main-model-js/model.json')
	predict()
	$('#file-input-container').show()
}

// For preview of image after browse
function display(input) {
	if (input.files && input.files[0]) {
		const reader = new FileReader()
		reader.onload = function(e) {
			$('#input-image').attr('src', e.target.result)
		}
		reader.readAsDataURL(input.files[0])
		// Add a little delay before calling the predict function
		setTimeout(function() {
			predict()
		}, 50)
		
	}
}
// When clicking the predict button
async function predict() {
	let input = document.getElementById('input-image')
	let step1 = tf.browser
		.fromPixels(input)
		.resizeNearestNeighbor([128, 128])
		.div(tf.scalar(255))
		.expandDims(0)
	const pred = await model.predict(step1).arraySync()
	const prob = tf.softmax(pred).arraySync()[0]
	displayResult(prob)
}

// Display the result
function displayResult(prob) {
	let htmlStr = ''
	let displayCounter = 0, maxToDisplay = MAX_CLASS_TO_DISPLAY
	const newResult = matchProbabilityAndClasses(prob)
	for (const key in newResult) {
		const percentDisplay = (newResult[key] * 100).toFixed(1)
		htmlStr += '\
			<div class="result-item-container"> \
				<div class="result-label">' + key + '</div> \
				<div class="result-bar"> \
					<div class="result-bar-value text-center" style="background: linear-gradient(90deg, #198754 ' + percentDisplay + '%, #FFFFFF ' + percentDisplay + '%);">' + percentDisplay + '%</div> \
				</div> \
			</div> \
		'
		displayCounter++
		if (displayCounter >= maxToDisplay) {
			break
		}
	}
	$('#results').html(htmlStr)
}

// Match probability with all classes
function matchProbabilityAndClasses(prob) {
	const allProbData = []
	for (let x=0; x<prob.length; x++) {
		allProbData[allClasses[x]] = prob[x]
	}
	return sortObjectbyValue(allProbData, false)
}

function sortObjectbyValue(obj={},asc=true){ 
	const ret = {}
	Object.keys(obj).sort((a,b) => obj[asc?a:b]-obj[asc?b:a]).forEach(s => ret[s] = obj[s])
	return ret
}