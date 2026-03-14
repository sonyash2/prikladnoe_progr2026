CXX = g++
CXXFLAGS = -std=c++11 -O3 -fopenmp
TARGET1 = matrix_mult_omp
TARGET2 = benchmark_omp
SRC1 = main_omp.cpp
SRC2 = benchmark_omp.cpp

all: $(TARGET1) $(TARGET2)

$(TARGET1): $(SRC1)
	$(CXX) $(CXXFLAGS) -o $(TARGET1) $(SRC1)

$(TARGET2): $(SRC2)
	$(CXX) $(CXXFLAGS) -o $(TARGET2) $(SRC2)

clean:
	rm -f $(TARGET1) $(TARGET2) *.txt

benchmark: $(TARGET2)
	./$(TARGET2)

.PHONY: all clean benchmark