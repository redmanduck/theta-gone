//init worker thread
var worker = new Worker('worker/worker.js');
var i = 0;

//wait for periodic info from workers
worker.onmessage = function(e) {
	$('#err').html(e.data.err);
	$('#iter').html(e.data.iter);
	$('#result').html(e.data.result);
	i++;
}