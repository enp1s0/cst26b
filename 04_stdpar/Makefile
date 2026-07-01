CXX=nvc++
CXXFLAGS=-std=c++17 -Wall -stdpar=gpu

TARGET=stdpar_sample.test

$(TARGET):stdpar_sample.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)
  
clean:
	rm -f $(TARGET)
