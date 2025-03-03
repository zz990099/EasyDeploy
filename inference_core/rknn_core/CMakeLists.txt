cmake_minimum_required(VERSION 3.0.2)
project(rknn_core)

add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)

find_package(glog REQUIRED)

set(source_file src/rknn_core.cpp
                src/rknn_core_factory.cpp)

add_library(${PROJECT_NAME} SHARED ${source_file})  

include_directories(
  include
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  glog::glog
  deploy_core
  rknnrt
)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)