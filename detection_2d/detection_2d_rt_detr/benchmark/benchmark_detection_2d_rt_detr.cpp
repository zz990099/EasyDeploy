#include <gtest/gtest.h>

#include "detection_2d_util/detection_2d_util.hpp"
#include "detection_2d_rt_detr/rt_detr.hpp"
#include "benchmark_utils/detection_2d_benchmark_utils.hpp"

using namespace easy_deploy;

#ifdef ENABLE_TENSORRT

#include "trt_core/trt_core.hpp"

std::shared_ptr<BaseDetectionModel> CreateRTDetrTensorRTModel()
{
  std::string                    model_path   = "/workspace/models/rt_detr_v2_single_input.engine";
  const int                      input_height = 640;
  const int                      input_width  = 640;
  const int                      input_channels    = 3;
  const int                      cls_number        = 80;
  const std::vector<std::string> input_blobs_name  = {"images"};
  const std::vector<std::string> output_blobs_name = {"labels", "boxes", "scores"};

  auto infer_core = CreateTrtInferCore(model_path);
  auto preprocess = CreateCudaDetPreProcess();

  auto rt_detr_model =
      CreateRTDetrDetectionModel(infer_core, preprocess, input_height, input_width, input_channels,
                                 cls_number, input_blobs_name, output_blobs_name);
  return rt_detr_model;
}

static void benchmark_detection_2d_rt_detr_tensorrt_sync(benchmark::State &state)
{
  benchmark_detection_2d_sync(state, CreateRTDetrTensorRTModel());
}
static void benchmark_detection_2d_rt_detr_tensorrt_async(benchmark::State &state)
{
  benchmark_detection_2d_async(state, CreateRTDetrTensorRTModel());
}
BENCHMARK(benchmark_detection_2d_rt_detr_tensorrt_sync)->Arg(500)->UseRealTime();
BENCHMARK(benchmark_detection_2d_rt_detr_tensorrt_async)->Arg(500)->UseRealTime();

#endif

#ifdef ENABLE_ORT

#include "ort_core/ort_core.hpp"

std::shared_ptr<BaseDetectionModel> CreateRTDetrOnnxRuntimeModel()
{
  std::string                    model_path     = "/workspace/models/rt_detr_v2_single_input.onnx";
  const int                      input_height   = 640;
  const int                      input_width    = 640;
  const int                      input_channels = 3;
  const int                      cls_number     = 80;
  const std::vector<std::string> input_blobs_name  = {"images"};
  const std::vector<std::string> output_blobs_name = {"labels", "boxes", "scores"};

  auto infer_core = CreateOrtInferCore(model_path);
  auto preprocess = CreateCpuDetPreProcess({0, 0, 0}, {255, 255, 255}, true, true);

  auto rt_detr_model =
      CreateRTDetrDetectionModel(infer_core, preprocess, input_height, input_width, input_channels,
                                 cls_number, input_blobs_name, output_blobs_name);
  return rt_detr_model;
}

static void benchmark_detection_2d_rt_detr_onnxruntime_sync(benchmark::State &state)
{
  benchmark_detection_2d_sync(state, CreateRTDetrOnnxRuntimeModel());
}
static void benchmark_detection_2d_rt_detr_onnxruntime_async(benchmark::State &state)
{
  benchmark_detection_2d_async(state, CreateRTDetrOnnxRuntimeModel());
}
BENCHMARK(benchmark_detection_2d_rt_detr_onnxruntime_sync)->Arg(100)->UseRealTime();
BENCHMARK(benchmark_detection_2d_rt_detr_onnxruntime_async)->Arg(100)->UseRealTime();

#endif

BENCHMARK_MAIN();
