#include <iostream>
#include <cstdlib> 
#include <stdint.h>
#include <string.h>
#include <armadillo>

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
	mat W0 = randu(2,4);
	mat W1 = randu(4,1);

	mat L0, L1, L2;

	for(int i = 0; i < 10000; i++){
		L0 = X;
		L1 = sgm(L0*W0);
		L2 = sgm(L1*W1);
		mat L2_error = Y - L2;
		mat L2_delta = L2_error % sgm_prime(L2);
		mat L1_error = L2_delta * trans(W1);
		mat L1_delta = L1_error % sgm_prime(L1);

		W1 = W1 + trans(L1)*L2_delta;
    	W0 = W0 + trans(L0)*L1_delta;


	}
	L2.print();

}