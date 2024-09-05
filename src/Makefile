.PHONY: all build install uninstall dvi dist tests gcov_report clang cppcheck leaks valgrind clean test
CC = g++ -std=c++17 -O3
CFLAGS = -Wall -Werror -Wextra
CPPCHECKFLAGS = --enable=all --language=c++ --std=c++17 --suppress=missingIncludeSystem --suppress=unusedFunction --suppress=passedByValue --suppress=unmatchedSuppression --suppress=missingInclude --suppress=unusedStructMember
PROJECT_NAME = MLP

SOURCES :=main.cpp \
        controller/*.cc \
        model/*.cc \
        view/*.cpp

SOURCES_H :=controller/*.h \
		model/*.h \
		view/*.h

MODEL_C := model/col_vector.cc \
		model/graph_layer.cc \
		model/matrix_layer.cc \
		model/matrix.cc \
		model/network_service.cc \
		model/neural_network.cc \
		model/neuron.cc \

OS = $(shell uname)
ifeq ($(OS),Darwin)
	CHECKFLAGS = -lgtest
	GCOVFLAGS = --coverage
else
	CHECKFLAGS = -lgtest -lpthread -lrt -lm
	GCOVFLAGS = -fprofile-arcs -ftest-coverage
endif

all: clean build

build:
	@rm -rf ./build
	@mkdir -p ./build/
	cmake -S . -B ./build
	cmake --build ./build

install: clean uninstall build
	mkdir -p $(HOME)/MLP/
	mkdir -p $(HOME)/MLP/resources/datasets
	mkdir -p $(HOME)/MLP/resources/weigths
	mkdir -p $(HOME)/MLP/resources/data-samples
	@cp -rf ./build/MLP.app $(HOME)/MLP/
	@cp -rf ./resources/datasets $(HOME)/MLP/resources
	@cp -rf ./resources/weigths $(HOME)/MLP/resources
	@cp -rf ./resources/data-samples $(HOME)/MLP/resources
	make clean
	@open $(HOME)/MLP/

uninstall:
	rm -rf $(HOME)/MLP/

tests:
	$(CC) $(CFLAGS) test.cc $(MODEL_C) $(CHECKFLAGS) -o test.out
	./test.out

clang:
	@cp ../materials/linters/.clang-format ./
	@clang-format -i controller/*
	@clang-format -i model/*
	@clang-format -i view/*.cpp
	@clang-format -i view/*.h
	@clang-format -i main.cpp
	@clang-format -i test.cc

	clang-format -n controller/*
	clang-format -n model/*
	clang-format -n view/*.cpp
	clang-format -n view/*.h
	clang-format -n main.cpp
	clang-format -n test.cc

	@rm .clang-format

cppcheck:
	@cppcheck $(CPPCHECKFLAGS) $(SOURCES)
	@cppcheck $(CPPCHECKFLAGS) $(SOURCES_H)
	@cppcheck $(CPPCHECKFLAGS) main.cpp
	@cppcheck $(CPPCHECKFLAGS) test.cc

leaks:
	CK_FORK=no leaks --atExit -- ./test.out

valgrind:
	valgrind --leak-check=full --show-leak-kinds=all --trace-children=yes ./test.out

clean:
	rm -f *.o *.a *.gcno *.gcda *.info *.out
	rm -rf ./report/
	rm -rf ./build/
	rm -rf .cmake
	rm -rf .qtc_clangd
