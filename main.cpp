#include <iostream>
#include <cstdlib> 
#include <stdint.h>
#include <string.h>
#include <armadillo>

#define HIDDEN_LAYERS 3
#define HIDDEN_LAYER_SIZE 5
#define INPUT_LAYER_SIZE 3
#define LEARNING_COEFF 0.5
#define DESIRED_ERROR_PERCENT 0.5
#define MAX_ITERATION_LEARN 55555

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
	mat X =  { {0,0,1}, {1,1,1}, {1,0,1}, {0,1,1} };
	mat _Y =  {0,1,1,0};
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
			Layers[j] = sgm(Layers[j-1] * Weights[j-1]);
		}

		// compute error for network
		mat Ll = Layers[Layers.size() - 1];
		mat Ll_error = Y - Ll;

		//TODO: Don't use max! 
		double err = as_scalar(max(Ll_error*100));
		cout << "Current Error: " << err << "%" << endl;
		if(err <= DESIRED_ERROR_PERCENT){
			cout <<  "DESIRED_ERROR_PERCENT reached at : " << err << '%' << endl;
			break;
		}
		if(i == MAX_ITERATION_LEARN - 1){
			cout << "MAX_ITERATION_LEARN reached at : " << err << '%' << endl;
		}

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