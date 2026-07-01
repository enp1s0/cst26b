NVCC ?= nvcc
TARGET := cublas_sample

.PHONY: all run clean

all: $(TARGET)

$(TARGET): cublas_sample.cu
	$(NVCC) -o $@ $< -lcublas

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
