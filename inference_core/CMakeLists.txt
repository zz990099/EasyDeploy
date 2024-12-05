cmake_minimum_required(VERSION 3.8)
project(inference_core)

if (ENABLE_ORT)
  add_subdirectory(ort_core)
endif()

if (ENABLE_TENSORRT)
  add_subdirectory(trt_core)
endif()

if (ENABLE_RKNN)
  add_subdirectory(rknn_core)
endif()