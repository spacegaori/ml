#!/bin/sh

set -xe

clang++ -std=c++2b -pedantic-errors -Wall -Weffc++ -Wextra -Wsign-conversion -o nn nn.cpp
