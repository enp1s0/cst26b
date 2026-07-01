CXX=nvcc
CXXFLAGS=-std=c++17 -arch=sm_100 -Xcompiler=-fopenmp

TARGET=01_vectoradd.test 02_innerproduct.test

all: ${TARGET}

%.test:%.cu
	$(CXX) $< -o $@ $(CXXFLAGS)

clean:
	rm -f $(TARGET)
