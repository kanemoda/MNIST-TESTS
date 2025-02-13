# FNN_MNIST/Makefile

# Compiler and flags
CXX       := g++
CXXFLAGS  := -Wall -std=c++14 -O2 -fopenmp -Iinclude -I/usr/include/eigen3

# Directories
SRCDIR    := src
OBJDIR    := obj
BINDIR    := .

# Find all source files and convert them to object files
SOURCES   := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS   := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SOURCES))

# Final executable name
TARGET    := fnn_project

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/$(TARGET) $(OBJECTS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR)/*.o $(TARGET)

run: $(TARGET)
	./$(TARGET)
