# Compiler and flags
CXX      = g++
CXXFLAGS = -std=c++11 -pthread -O2 -Wall -Wextra -Iinclude -mavx -fopenmp

# Directories
SRCDIR = src
OBJDIR = obj
BINDIR = .

# Find all source files in the src directory
SRCS := $(wildcard $(SRCDIR)/*.cpp)
# Replace src/ with obj/ and .cpp with .o to create object file list
OBJS := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCS))

# Name of the final executable
TARGET = mnist_project

.PHONY: all clean run

# Default target: build the executable
all: $(TARGET)

# Link all object files into the final executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/$@ $(OBJS)

# Compile source files into object files in the obj directory
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -rf $(OBJDIR) $(TARGET)

# Build (if needed) and run the executable
run: $(TARGET)
	./$(BINDIR)/$(TARGET)
