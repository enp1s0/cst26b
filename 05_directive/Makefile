CXX=nvc++
CXXFLAGS=-std=c++17 -Wall -gpu=mem:managed

TARGET=openacc_sample.test openmp_offload_sample.test

all: ${TARGET}

openacc_sample.test:01_openacc.cpp
	$(CXX) $< -o $@ $(CXXFLAGS) -acc -Minfo=acce

openmp_offload_sample.test:02_openmp_offload.cpp
	$(CXX) $< -o $@ $(CXXFLAGS) -mp=gpu -Minfo=mp
  
clean:
	rm -f $(TARGET)
