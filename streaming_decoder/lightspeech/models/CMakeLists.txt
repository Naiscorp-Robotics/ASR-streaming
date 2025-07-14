cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(MyTorchModel)

set(CMAKE_PREFIX_PATH "/home/jason/miniconda3/envs/lightspeech/lib/cmake/Torch")

find_package(Torch REQUIRED)

add_executable(main inference.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)