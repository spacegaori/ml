#!/bin/sh

set -xe

clang++ -std=c++2b -pedantic-errors -Wall -Weffc++ -Wextra -Wsign-conversion -o twice twice.cpp
clang++ -std=c++2b -pedantic-errors -Wall -Weffc++ -Wextra -Wsign-conversion -o gates gates.cpp
clang++ -std=c++2b -pedantic-errors -Wall -Weffc++ -Wextra -Wsign-conversion -o xor xor.cpp
