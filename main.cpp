#include <iostream>
#include <cstdlib> 
#include <stdint.h>
#include <string.h>
#include <armadillo>

#define HIDDEN_LAYERS 2
#define HIDDEN_LAYER_SIZE 5
#define INPUT_LAYER_SIZE 2
#define LEARNING_COEFF 0.001
#define DESIRED_ERROR_PERCENT 0.05
#define MAX_ITERATION_LEARN 5000000

using namespace std;
using namespace arma;

mat actv(const mat& x){
	return 1/(1+exp(x*-1));
}

mat actv_prime(mat& x){
	return x % (1-x);
}

void seed(mat& X, mat& _Y, int seed_size){
	X = mat(seed_size, INPUT_LAYER_SIZE);
	_Y = mat(1,seed_size);

	for(int s = 0; s < seed_size; s++){
		X(s,0) = s+1;
		X(s,1) = s+2;
		X(s,2) = s+3;

		_Y(0, s) = (s+1) + (s+2)*4 + (s+3)*3;
	}

	X.print();
	_Y.print();
}

int main(int argc, char *argv[]){


	//Training Set
	mat X =  { {1, 1}, {2, 1}, {5,1}, {6, 1}, {10,1}};
	mat _Y =  {1/100.0, 4/100.0, 25/100.0, 36/100.0, 100/100.0};


	//seed(X, _Y, 10);

	mat Y = trans(_Y);
	// Weight matrices
	mat W0 = randu(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
	mat Wl = randu(HIDDEN_LAYER_SIZE,1);

	vector<mat> Weights = { W0 };
	for(int l = 0; l < HIDDEN_LAYERS - 1; l++){
		mat whdn = randu(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE);
		Weights.push_back(whdn);
	}
	Weights.push_back(Wl);

	//Layer matrices
	mat Ll;
	vector<mat> Layers = { X };
	for(int l = 0; l < HIDDEN_LAYERS; l++){
		mat hdn = mat(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE);
		Layers.push_back(hdn);
	}
	Layers.push_back(Ll);

	cout << "Training .." << endl;
	//Backpropagation
	for(int i = 0; i < MAX_ITERATION_LEARN; i++){
		//compute output of entire network
		for(int j = 1; j < Layers.size(); j++){
			Layers[j] = actv(Layers[j-1] * Weights[j-1]);
		}

		// compute error for network
		mat Ll = Layers[Layers.size() - 1];
		mat Ll_error = Y - Ll;

		//TODO: Don't use max! 
		double err = abs(as_scalar(max(Ll_error*100)));
		if(i % 20000 == 0){
			cout << "Iteration " << i << " Error : " << endl;
			// Ll_error.print();
		}
		if(err <= DESIRED_ERROR_PERCENT){
			cout <<  "DESIRED_ERROR_PERCENT reached at : ";
			Ll_error.print();
			cout << endl;
			break;
		}
		if(i == MAX_ITERATION_LEARN - 1){
			cout << "MAX_ITERATION_LEARN reached at : ";
			Ll_error.print();
			cout << endl;
			break;
		}

		mat Ll_delta = Ll_error % actv_prime(Ll);
		// cout << "Ll_delta : ";
		// Ll_delta.print();
		//update last weight 
		int w_idx = Weights.size() - 1;
		Weights[w_idx] = Weights[w_idx] + LEARNING_COEFF*trans(Layers[w_idx])*Ll_delta;

		mat Lprev_delta = Ll_delta;
		// propagate backward for the rest of the network
		for(int j = Weights.size() - 2; j >= 0; j--){
			mat Lj_error = Lprev_delta * trans(Weights[j + 1]);
			mat Lj_delta = Lj_error % actv_prime(Layers[j + 1]);
			//Update weight
			Weights[j] = Weights[j] + LEARNING_COEFF*trans(Layers[j])*Lj_delta;

			Lprev_delta = Lj_delta;
		}
	}

	Layers[Layers.size() - 1].print();

	Layers[0] = { {51, 1} };
	//Compute unseen
	for(int j = 1; j < Layers.size(); j++){
			Layers[j] = actv(Layers[j-1] * Weights[j-1]);
	}
	cout << "Prediction for X = ";
	Layers[0].print();
	Layers[Layers.size() - 1].print();

}