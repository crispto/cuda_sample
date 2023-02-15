CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin g++
GCC_FLAGS := --std=c++11 --threads 0
INCLUDES  := -I../../common/inc

all: build

build: main

main: main.cu
	$(NVCC) $(INCLUDES) $(GCC_FLAGS) -o $@ $^ 

clean:
	rm -rf main