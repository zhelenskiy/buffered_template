cmake_minimum_required(VERSION 3.15)
project(buffered_template)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS -O3)

add_executable(buffered_template main.cpp lambdas.h)