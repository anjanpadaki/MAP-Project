# P4 - CUDA GEMM and Cluster Integration
#   make            build everything (needs nvcc + g++)
#   make cpu        CPU-only targets (no CUDA needed)
#   make ARCH=sm_75 override the GPU architecture if -arch=native is unsupported
#   make NVML=1     also record GPU power draw (links libnvidia-ml)

CXX      ?= g++
CC       ?= gcc
NVCC     ?= nvcc
ARCH     ?= native

CXXFLAGS ?= -O3 -march=native -std=c++14 -Wall -Wextra
CFLAGS   ?= -O3 -march=native -fopenmp -Wall
NVFLAGS  ?= -O3 -std=c++14 -arch=$(ARCH) -lineinfo -Xcompiler -Wall
NVLIBS   := -lcublas

ifeq ($(NVML),1)
NVFLAGS  += -DUSE_NVML
NVLIBS   += -lnvidia-ml
endif

BIN := bin

all: cpu gpu
cpu: $(BIN)/gemm_cpu $(BIN)/stream_cpu
gpu: $(BIN)/gemm_gpu $(BIN)/gpu_stream

$(BIN):
	mkdir -p $(BIN)

$(BIN)/gemm_cpu: src/gemm_cpu.cpp src/common.h | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN)/stream_cpu: src/stream_cpu.c | $(BIN)
	$(CC) $(CFLAGS) $< -o $@ -lm

$(BIN)/gemm_gpu: src/gemm_gpu.cu src/common.h | $(BIN)
	$(NVCC) $(NVFLAGS) $< -o $@ $(NVLIBS)

$(BIN)/gpu_stream: src/gpu_stream.cu | $(BIN)
	$(NVCC) $(NVFLAGS) $< -o $@

# Records the exact compiler flags used, for the hardware section of the report.
flags:
	@echo "CXX=$(CXX) CXXFLAGS=$(CXXFLAGS)"
	@echo "CC=$(CC) CFLAGS=$(CFLAGS)"
	@echo "NVCC=$(NVCC) NVFLAGS=$(NVFLAGS) NVLIBS=$(NVLIBS)"

clean:
	rm -rf $(BIN)

.PHONY: all cpu gpu flags clean
