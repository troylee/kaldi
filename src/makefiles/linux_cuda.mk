
CUDA_INCLUDE= -I$(CUDATKDIR)/include
CUDA_FLAGS = -g -Xcompiler -fPIC --verbose --machine 32 -DHAVE_CUDA
CUDA_FLAGS += --gpu-architecture compute_13 --gpu-code sm_13

CXXFLAGS += -DHAVE_CUDA -I$(CUDATKDIR)/include 
CUDA_LDFLAGS += -L$(CUDATKDIR)/lib -Wl,-rpath,$(CUDATKDIR)/lib
CUDA_LDLIBS += -lcublas -lcudart -lcuda

