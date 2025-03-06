# CMakeList.txt : CMake project for Neural Network, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8) 




# 14.29.30133 doesnt work
set(CMAKE_C_COMPILER "E:/vs2019/IDE/VC/Tools/MSVC/14.29.30037/bin/HostX64/x64/cl.exe")
set(CMAKE_CXX_COMPILER "E:/vs2019/IDE/VC/Tools/MSVC/14.29.30037/bin/HostX64/x64/cl.exe")
project ("Neural-Network-AI")
# Add source to this project's executable.
add_executable (CMakeTarget "Neural Network.cpp" "Neural Network.h" ) 

# TODO: Add tests and install targets if needed.
