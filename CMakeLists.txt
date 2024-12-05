cmake_minimum_required(VERSION 3.8)
project(easy_deployment)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 

add_subdirectory(deploy_core)
add_subdirectory(deploy_utils)


add_subdirectory(inference_core)
add_subdirectory(detection_2d)

add_subdirectory(sam)

if (BUILD_TESTING)
  add_subdirectory(simple_tests)
endif()