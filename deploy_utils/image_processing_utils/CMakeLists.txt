cmake_minimum_required(VERSION 3.8)
project(image_processing_utils)

add_compile_options(-std=c++17)
add_compile_options(-O3)
set(CMAKE_CXX_STANDARD 17)

if(ENABLE_TENSORRT)
  find_package(CUDA REQUIRED)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
  enable_language(CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
endif()

find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)

set(source_file 
    src/detection_2d_preprocess_cpu.cpp
    src/detection_2d_postprocess_cpu.cpp
)

if(ENABLE_TENSORRT)
  list(APPEND source_file 
          src/detection_2d_preprocess_cuda.cpp
          src/detection_2d_cuda_preprocess.cu
  )
endif()

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED ${source_file})   

target_link_libraries(${PROJECT_NAME} PUBLIC 
  glog::glog
  ${OpenCV_LIBS}
  deploy_core
)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)