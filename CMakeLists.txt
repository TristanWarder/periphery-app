cmake_minimum_required(VERSION 3.1)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# Name of the project (will be the name of the plugin)
project(yolov8-runner LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

# TensorRT
set(TensorRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu)
set(TensorRT_LIBRARIES /usr/lib/x86_64-linux-gnu)

message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")

# Build a shared library named after the project from the files in `src/`
file(GLOB SOURCE_FILES "src/*.cc" "src/*.h")
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})

# Gives our library file a .node extension without any "lib" prefix
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

# Essential include files to build a node addon,
# You should add this line in every CMake.js based project
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_JS_INC} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} include)

# Essential library files to link to a node addon
# You should add this line in every CMake.js based project
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB} nvinfer nvinfer_plugin nvonnxparser ${CUDA_LIBRARIES} ${OpenCV_LIBS})
