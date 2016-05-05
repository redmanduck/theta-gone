#include <iostream>
#include <cstdlib> 
#include <stdint.h>
#include <string.h>
#include <armadillo>

#define LEARNING_COEFF 1
#define HIDDEN_LAYERS 3
#define HIDDEN_LAYER_SIZE 4
#define INPUT_LAYER_SIZE 2

using namespace std;
using namespace arma;

mat sgm(const mat& x){
	return 1/(1+exp(x*-1));
}

mat sgm_prime(const mat& x){
	return x % (1-x);
}

int main(int argc, char *argv[]){
	//Training Set
	mat X =  { {0,1}, {1,1}, {0,0}, {1,0} };
	mat _Y =  {1,0,0,1};
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

	//Backpropagation
	for(int i = 0; i < 10000; i++){
		//compute output of entire network
		for(int j = 1; j < Layers.size(); j++){
			Layers[j] = sgm(Layers[j-1] * Weights[j-1]);
		}

		// compute error for network
		mat Ll = Layers[Layers.size() - 1];
		mat Ll_error = Y - Ll;
		mat Ll_delta = Ll_error % sgm_prime(Ll);
		//update last weight 
		int w_idx = Weights.size() - 1;
		Weights[w_idx] = Weights[w_idx] + LEARNING_COEFF*trans(Layers[w_idx])*Ll_delta;

		mat Lprev_delta = Ll_delta;
		// propagate backward for the rest of the network
		for(int j = Weights.size() - 2; j >= 0; j--){
			mat Lj_error = Lprev_delta * trans(Weights[j + 1]);
			mat Lj_delta = Lj_error % sgm_prime(Layers[j + 1]);
			//Update weight
			Weights[j] = Weights[j] + LEARNING_COEFF*trans(Layers[j])*Lj_delta;

			Lprev_delta = Lj_delta;
		}
	}

	Layers[Layers.size() - 1].print();

}