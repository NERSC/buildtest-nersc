NVCC_OPTS ?= -g -arch=sm_80

all: main

main:
	nvcc -x cu ${NVCC_OPTS} main.cpp

clean:
	-rm -f a.out
