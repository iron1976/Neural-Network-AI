# CMakeList.txt : CMake project for Neural Network, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Neural Network")

# Add source to this project's executable.
add_executable (CMakeTarget "Neural Network.cpp" "Neural Network.h" )
include_directories(${YOUR_DIRECTORY}/Dependencies/glmsaddass)

# TODO: Add tests and install targets if needed.
