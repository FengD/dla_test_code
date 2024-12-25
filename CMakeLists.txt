cmake_minimum_required(VERSION 3.10)
project(TensorRT_Inference)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

set(TENSORRT_LIB_DIR /usr/local/cuda-12.2/targets/aarch64-linux/lib) # 根据实际路径修改
link_directories(${TENSORRT_LIB_DIR})

add_executable(TensorRT_Inference test.cpp)

target_link_libraries(TensorRT_Inference nvinfer nvinfer_plugin cudart ${OpenCV_LIBS})
