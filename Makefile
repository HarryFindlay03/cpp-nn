
default: cpp_nn

cpp_nn:
	g++-13 -I "./include/" src/cpp_nn.cpp -o bin/cpp_nn && ./bin/cpp_nn
