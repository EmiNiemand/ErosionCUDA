cmake_minimum_required(VERSION 3.25)
project(CUDA)

set(CMAKE_CXX_STANDARD 23)

add_executable(CUDA main.cpp)

add_subdirectory(thirdparty)