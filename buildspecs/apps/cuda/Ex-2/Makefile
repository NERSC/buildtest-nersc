NVCCFLAGS = -arch=sm_80

vec_add: kernels.o kernels.h vecAdd.cpp
	g++ -o $@ -L$(CUDA_HOME)/lib64 -lcudart vecAdd.cpp kernels.o
kernels.o: kernels.cu kernels.h
	nvcc $(NVCCFLAGS) -c kernels.cu -o $@
clean:
	rm *.o vec_add

