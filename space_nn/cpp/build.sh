#!/bin/sh

set -xe

clang++ -std=c++2b -g -pedantic-errors -Wall -Weffc++ -Wextra -Wsign-conversion -o nn nn.cpp
