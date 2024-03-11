CC = g++-13
CC_FLAGS = -I include/

default: cpp_nn

network.o:
	$(CC) $(CC_FLAGS) -c src/network.cpp -o build/network.o

cpp_nn: src/cpp_nn.cpp network.o
	$(CC) $(CC_FLAGS) src/cpp_nn.cpp  build/network.o -o bin/cpp_nn
