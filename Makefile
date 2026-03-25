# This is your make file
# You may change it and we use it to build your code.
# DO NOT CHANGE RECIPE FOR TEST RELATED TARGETS 
CXX := g++
NVCC := nvcc
CXXFLAGS := -std=c++17 -Wall -Wextra -pedantic -MMD -MP
NVCCFLAGS := -std=c++17 -arch=native -Xcompiler -Wall
LDFLAGS := -lcudart
BUILD ?= release

ifeq ($(BUILD),debug)
  CXXFLAGS += -g -O0
  NVCCFLAGS += -g -G -O0
else
  CXXFLAGS += -O2
  NVCCFLAGS += -O2
endif

SRC_DIR := src
INC_DIR := include
KERNEL_INC_DIR := kernel
BUILD_DIR := build
BIN_DIR := bin
BENCH_DIR := benchmarks
TARGET := llm

CPP_SOURCES := main.cpp $(SRC_DIR)/tokenizer_bpe.cpp $(SRC_DIR)/loader.cpp
CPP_OBJECTS := $(BUILD_DIR)/main.o $(BUILD_DIR)/tokenizer_bpe.o $(BUILD_DIR)/loader.o

CUDA_SOURCES := $(wildcard $(SRC_DIR)/*.cu)
CUDA_OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SOURCES))

OBJECTS := $(CPP_OBJECTS) $(CUDA_OBJECTS)

DEPS := $(OBJECTS:.o=.d)
INCLUDES := -I$(INC_DIR) -I. -I$(KERNEL_INC_DIR) -I/usr/local/cuda/include


all: $(BIN_DIR)/$(TARGET)
$(BIN_DIR)/$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(NVCC) $(OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/main.o: main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/tokenizer_bpe.o: $(SRC_DIR)/tokenizer_bpe.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/loader.o: $(SRC_DIR)/loader.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

.PHONY: clean

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	rm -f a.out

.PHONY: run

run: all
	./$(BIN_DIR)/$(TARGET)
-include $(DEPS)

# ------------------------------------------------------------
# Tests build
# Don't forget to add your required objects as well.

.PHONY: tests

TEST_OBJECTS := $(BUILD_DIR)/test.o $(BUILD_DIR)/test_api.o $(BUILD_DIR)/tokenizer_bpe.o $(BUILD_DIR)/loader.o $(CUDA_OBJECTS)

tests: $(BIN_DIR)/tests

$(BIN_DIR)/tests: $(TEST_OBJECTS) | $(BIN_DIR)
	$(NVCC) $(TEST_OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test.o: tests/test.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/test_api.o: tests/test_api.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# ------------------------------------------------------------
# Benchmarks build
BENCH_SOURCES := $(wildcard $(BENCH_DIR)/*.cu)
BENCH_TARGETS := $(patsubst $(BENCH_DIR)/%.cu,$(BIN_DIR)/%,$(BENCH_SOURCES))
BENCH_OBJECTS := $(patsubst $(BENCH_DIR)/%.cu,$(BUILD_DIR)/bench_%.o,$(BENCH_SOURCES))

.PHONY: benchmarks
benchmarks: $(BENCH_TARGETS)

# Link each benchmark binary against the shared CUDA objects
$(BIN_DIR)/%: $(BUILD_DIR)/bench_%.o $(CUDA_OBJECTS) | $(BIN_DIR)
	$(NVCC) $^ -o $@ $(LDFLAGS)

# Compile each benchmark source
$(BUILD_DIR)/bench_%.o: $(BENCH_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

.PHONY: run_benchmarks
run_benchmarks: benchmarks
	@for b in $(BENCH_TARGETS); do \
		echo "Running $$b ..."; \
		./$$b; \
		echo ""; \
	done