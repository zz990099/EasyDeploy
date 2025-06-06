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

find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV REQUIRED)

set(source_file
  test_detection_2d_rt_detr.cpp
)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(test_detection_2d_rt_detr ${source_file})

target_link_libraries(test_detection_2d_rt_detr PUBLIC
  GTest::gtest_main
  glog::glog
  ${OpenCV_LIBS}
  deploy_core
  image_processing_utils
  detection_2d_rt_detr
  test_utils
  ${platform_core_packages}
)

if(ENABLE_TENSORRT)
  target_compile_definitions(test_detection_2d_rt_detr PRIVATE ENABLE_TENSORRT)
endif()

if(ENABLE_RKNN)
  target_compile_definitions(test_detection_2d_rt_detr PRIVATE ENABLE_RKNN)
endif()

if(ENABLE_ORT)
  target_compile_definitions(test_detection_2d_rt_detr PRIVATE ENABLE_ORT)
endif()

gtest_discover_tests(test_detection_2d_rt_detr)
