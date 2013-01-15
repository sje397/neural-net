#include "nngraddesc.h"

#include <iostream>

int main(int /*argc*/, char *argv[])
{
	srand(time(0));

	// test using xor
	typedef sje::BackPropagationNeuralNet<2, 3, 1> Net;
	Net nn;

	Net::vec in(2), out(1);
	for(int i = 0; i < 4; i++) {
		in(0) = i & 1;
		in(1) = (i >> 1) & 1;
		out(0) = int(in(0)) ^ int(in(1));
		nn.train(in, out);
	}

	nn.desc();

	Net::State state;
	for(int i = 0; i < 4; i++) {
		in(0) = i & 1;
		in(1) = (i >> 1) & 1;
		out = nn.calc(in, state);
		std::cout << "output: " << out(0) << ", expected: " << (int(in(0)) ^ int(in(1))) << std::endl;
	}

	return 0;
}
