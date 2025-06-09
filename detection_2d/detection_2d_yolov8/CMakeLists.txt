cmake_minimum_required(VERSION 3.8)
project(detection_2d_yolov8)

add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

set(source_file src/yolov8.cpp
                src/yolov8_factory.cpp)

add_library(${PROJECT_NAME} SHARED ${source_file})

target_link_libraries(${PROJECT_NAME} PUBLIC
  ${OpenCV_LIBS}
  deploy_core
  common_utils
)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

if (BUILD_TESTING)
  add_subdirectory(test)
endif()

if (BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

if (BUILD_EVAL)
  add_subdirectory(eval)
endif()
