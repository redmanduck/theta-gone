var math = require('mathjs');
var sgm = function(x){
	return math.dotDivide(1, math.add(1,math.exp(math.multiply(-1, x)))  );
}

var sgm_prime = function(x){
	return math.dotMultiply(x, (math.subtract(1, x)));
}

//Define training set
var X =  math.matrix([[0,1],
         [1,1],
         [0,0],
         [1,0]])
var Y = math.transpose(math.matrix([[1,0,0,1]]));

// Weight matrices
var W0 = math.random([2,4], 0, 1);
var W1 = math.random([4,1], 0, 1);

//L0 = layer 0, L1 = layer 1, ..etc
var L0 = math.matrix([]);
var L1 = math.matrix([]);
var L2 = math.matrix([]);

for(var i = 0; i < 10000; i++){
	L0 = math.clone(X);
	L1 = sgm(math.multiply(L0,W0));
    L2 = sgm(math.multiply(L1,W1));
	var L2_error = math.subtract(Y, L2);
	var L2_delta = math.dotMultiply(L2_error, sgm_prime(L2));
	var L1_error = math.multiply(L2_delta, math.transpose(W1));
	var L1_delta = math.dotMultiply(L1_error, sgm_prime(L1));
    W1 = math.add(W1, math.multiply(math.transpose(L1), L2_delta));
    W0 = math.add(W0, math.multiply(math.transpose(L0), L1_delta));
}

//result
console.log(L2);
