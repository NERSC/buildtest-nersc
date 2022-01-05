NVCCFLAGS = -arch=sm_80

vec_add: vecAdd.cu
	nvcc $(NVCCFLAGS) vecAdd.cu -o $@
clean:
	rm vec_add

