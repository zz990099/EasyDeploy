cmake_minimum_required(VERSION 3.8)
project(ort_core)

add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)

find_package(glog REQUIRED)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  message(STATUS "Building OnnxRuntime-Core on x86_64 platform ...")
  set(ONNXRUNTIME_INCLUDE_DIRS onnxruntime-pkg-x86/include)
  FILE(GLOB ONNXRUNTIME_LIBS onnxruntime-pkg-x86/lib/*.so)  
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(STATUS "Building OnnxRuntime-Core on aarch64 platform ...")
  set(ONNXRUNTIME_INCLUDE_DIRS onnxruntime-pkg-aarch64/include)
  FILE(GLOB ONNXRUNTIME_LIBS onnxruntime-pkg-aarch64/lib/*.so)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}. \
                      Supported architectures are x86_64 and aarch64.")
endif()


set(source_file 
  src/ort_core.cpp
  src/ort_core_factory.cpp
)

include_directories(
  include
  ${ONNXRUNTIME_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED ${source_file})   

target_link_libraries(${PROJECT_NAME} PUBLIC
  ${ONNXRUNTIME_LIBS}
  glog::glog
  deploy_core
)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)