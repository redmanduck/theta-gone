importScripts('../node_modules/linear-algebra/dist/linear-algebra.min.js')

//global
var Vector = linearAlgebra().Vector,
	Matrix = linearAlgebra().Matrix,
	HIDDEN_LAYER_SIZE = 4,
	HIDDEN_LAYER = 2,
	LEARNING_COEFF = 0.001;

Matrix.prototype.cmul = function(m2) {
	var data = this.data,
		rows = this.rows,
		cols = this.cols,
		data2 = m2.data,
		rows2 = m2.rows,
		cols2 = m2.cols;

	var r, c, c2, res = new Array(rows);

	for(r = 0; r < rows; ++r) {
		res[r] = new Array(cols2);
		for(c = 0; c < cols2; ++c) {
			res[r][c] = 0;
			for(c2 = 0; c2 < cols; ++c2) {
				res[r][c] += data[r][c2] * data2[c2][c];
			}
		}
	}
	return new Matrix(res);
}


//sigmoid function
var sigmoid = {
	fn: function(m) {
		return m.sigmoid();
	},
	prime: function(m) {
		return m.eleMap(function(v) {
			return v * (1 - v);
		});
	}
};

//float random
var frand = function(v) {
	return Math.random()*2.0 - 1.0;
};

//choose activator
var Activator = sigmoid;

//training set
var X = (new Matrix([ 0.001, 0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.4, 0.7, 1.0 ])).trans();
var Y = X.eleMap(function(v) { return v*v*v });

//weights in-out
var Win = Matrix.zero(X.cols, HIDDEN_LAYER_SIZE).eleMap(frand);
var Wout = Matrix.zero(HIDDEN_LAYER_SIZE, Y.cols).eleMap(frand);

console.log(Win.data.toString());

//weights array with w hidden
var Weights = [ Win ];
for (var i = 0; i < HIDDEN_LAYER - 1; i++) {
	var Whdn = Matrix.zero(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE).eleMap(frand);
	Weights.push(Whdn);
};
Weights.push(Wout);

//layers
var Layers = [X];
for (var i = 0; i < HIDDEN_LAYER; i++) {
	var Mhdn = Matrix.zero(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE);
	Layers.push(Mhdn);
};
Layers.push(new Matrix([]));

var Lout, Lout_error, Lout_error_percent, err, Lout_delta, 
	w_idx, 
	Lj_error, Lj_delta, Lprev_delta;
var i = 0, 
	j = 0;

//training loop
while(true) {

	//compute output of entire net
	for (j = 1; j < Layers.length; ++j) {
		Layers[j] = Activator.fn( Layers[j-1].cmul(Weights[j-1]) );
	};
	//compute error for net
	Lout = Layers[Layers.length - 1];
	Lout_error = Lout.minus(Y);
	Lout_error_percent = Lout_error.div(Lout).mulEach(100);

	err = Lout_error_percent.eleMap(function(v) {
		return Math.abs(v);
	}).getSum() / (Lout_error_percent.rows * Lout_error_percent.cols);


	//print
	if(i % 100000 == 0) {

		//create html text
		var lis = '';
		for (var x = 0; x < X.data.length; x++) {
			lis += '<li>(' + X.data[x][0].toFixed(10).toString() + ')^3 = ' + Y.data[x][0].toFixed(10).toString() + ' ~~~ ' + Lout.data[x][0].toFixed(10).toString() + '</li>';
		};

		//send back some response
		postMessage({
			err: err,
			iter: i,
			result: lis
		});
	}

	//backpropagation
	Lout_delta = Lout_error.mul(Activator.prime(Lout));

	//output layer weight update
	w_idx = Weights.length - 1;
	Weights[w_idx] = Weights[w_idx].minus(Layers[w_idx].trans().cmul(Lout_delta).mulEach(LEARNING_COEFF));

	Lprev_delta = Lout_delta;
	//hidden layer weight update
	for (j = Weights.length - 2; j >= 0; --j) {
		Lj_error = Lprev_delta.cmul(Weights[j+1].trans());
		Lj_delta = Lj_error.mul(Activator.prime(Layers[j+1]));

		//update
		Weights[j] = Weights[j].minus(Layers[j].trans().cmul(Lj_delta).mulEach(LEARNING_COEFF));
		Lprev_delta = Lj_delta;
	};

	//increment
	i++;
}