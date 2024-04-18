CC=g++
INCLUDE_DIR=include/
CXXFLAGS=-I $(INCLUDE_DIR)

default: main

network.o:
	$(CC) $(CXXFLAGS) -c src/mlp-cpp/network.cpp -o build/$@

main: network.o
	$(CC) $(CXXFLAGS) src/main.cpp build/network.o -o bin/main

install:
	mkdir -p bin
	mkdir -p build