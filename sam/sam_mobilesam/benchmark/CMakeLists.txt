add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)

if(ENABLE_TENSORRT)
  list(APPEND platform_core_packages trt_core)
endif()

if(ENABLE_RKNN)
  list(APPEND platform_core_packages rknn_core)
endif()

if(ENABLE_ORT)
  list(APPEND platform_core_packages ort_core)
endif()

find_package(glog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(benchmark REQUIRED)

set(source_file
  benchmark_sam_mobilesam.cpp
)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(benchmark_sam_mobilesam ${source_file})

target_link_libraries(benchmark_sam_mobilesam PUBLIC
  benchmark::benchmark
  glog::glog
  ${OpenCV_LIBS}
  deploy_core
  image_processing_utils
  sam_mobilesam
  benchmark_utils
  ${platform_core_packages}
)

if(ENABLE_TENSORRT)
  target_compile_definitions(benchmark_sam_mobilesam PRIVATE ENABLE_TENSORRT)
endif()

if(ENABLE_RKNN)
  target_compile_definitions(benchmark_sam_mobilesam PRIVATE ENABLE_RKNN)
endif()

if(ENABLE_ORT)
  target_compile_definitions(benchmark_sam_mobilesam PRIVATE ENABLE_ORT)
endif()
