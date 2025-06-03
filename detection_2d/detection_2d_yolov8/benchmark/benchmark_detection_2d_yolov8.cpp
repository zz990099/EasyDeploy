#include <gtest/gtest.h>

#include "detection_2d_util/detection_2d_util.h"
#include "detection_2d_yolov8/yolov8.h"
#include "benchmark_utils/detection_2d_benchmark_utils.hpp"

using namespace inference_core;
using namespace detection_2d;
using namespace benchmark_utils;

#ifdef ENABLE_TENSORRT

#include "trt_core/trt_core.h"

std::shared_ptr<BaseDetectionModel> CreateYolov8TensorRTModel()
{
  std::string                    model_path        = "/workspace/models/yolov8n.engine";
  const int                      input_height      = 640;
  const int                      input_width       = 640;
  const int                      input_channels    = 3;
  const int                      cls_number        = 80;
  const std::vector<std::string> input_blobs_name  = {"images"};
  const std::vector<std::string> output_blobs_name = {"output0"};

  auto infer_core  = CreateTrtInferCore(model_path);
  auto preprocess  = CreateCudaDetPreProcess();
  auto postprocess = CreateYolov8PostProcessCpuOrigin(input_height, input_width, cls_number);

  auto yolov8_model =
      CreateYolov8DetectionModel(infer_core, preprocess, postprocess, input_height, input_width,
                                 input_channels, cls_number, input_blobs_name, output_blobs_name);
  return yolov8_model;
}

static void benchmark_detection_2d_yolov8_tensorrt_sync(benchmark::State &state)
{
  benchmark_detection_2d_sync(state, CreateYolov8TensorRTModel());
}
static void benchmark_detection_2d_yolov8_tensorrt_async(benchmark::State &state)
{
  benchmark_detection_2d_async(state, CreateYolov8TensorRTModel());
}
BENCHMARK(benchmark_detection_2d_yolov8_tensorrt_sync)->Arg(1000)->UseRealTime();
BENCHMARK(benchmark_detection_2d_yolov8_tensorrt_async)->Arg(1000)->UseRealTime();

#endif

#ifdef ENABLE_ORT

#include "ort_core/ort_core.h"

std::shared_ptr<BaseDetectionModel> CreateYolov8OnnxRuntimeModel()
{
  std::string                    model_path        = "/workspace/models/yolov8n.onnx";
  const int                      input_height      = 640;
  const int                      input_width       = 640;
  const int                      input_channels    = 3;
  const int                      cls_number        = 80;
  const std::vector<std::string> input_blobs_name  = {"images"};
  const std::vector<std::string> output_blobs_name = {"output0"};

  auto infer_core  = CreateOrtInferCore(model_path);
  auto preprocess  = CreateCpuDetPreProcess({0, 0, 0}, {255, 255, 255}, true, true);
  auto postprocess = CreateYolov8PostProcessCpuOrigin(input_height, input_width, cls_number);

  auto yolov8_model =
      CreateYolov8DetectionModel(infer_core, preprocess, postprocess, input_height, input_width,
                                 input_channels, cls_number, input_blobs_name, output_blobs_name);
  return yolov8_model;
}

static void benchmark_detection_2d_yolov8_onnxruntime_sync(benchmark::State &state)
{
  benchmark_detection_2d_sync(state, CreateYolov8OnnxRuntimeModel());
}
static void benchmark_detection_2d_yolov8_onnxruntime_async(benchmark::State &state)
{
  benchmark_detection_2d_async(state, CreateYolov8OnnxRuntimeModel());
}
BENCHMARK(benchmark_detection_2d_yolov8_onnxruntime_sync)->Arg(200)->UseRealTime();
BENCHMARK(benchmark_detection_2d_yolov8_onnxruntime_async)->Arg(200)->UseRealTime();

#endif

#ifdef ENABLE_RKNN

#include "rknn_core/rknn_core.h"

std::shared_ptr<BaseDetectionModel> CreateYolov8RknnModel()
{
  std::string                    model_path       = "/workspace/models/yolov8n_divide_opset11.rknn";
  const int                      input_height     = 640;
  const int                      input_width      = 640;
  const int                      input_channels   = 3;
  const int                      cls_number       = 80;
  const std::vector<std::string> input_blobs_name = {"images"};
  const std::vector<std::string> output_blobs_name = {"318", "onnx::ReduceSum_326", "331",
                                                      "338", "onnx::ReduceSum_346", "350",
                                                      "357", "onnx::ReduceSum_365", "369"};

  auto infer_core  = CreateRknnInferCore(model_path, {{"images", RknnInputTensorType::RK_UINT8}});
  auto preprocess  = CreateCpuDetPreProcess({0, 0, 0}, {1, 1, 1}, false, false);
  auto postprocess = CreateYolov8PostProcessCpuDivide(input_height, input_width, cls_number);

  auto yolov8_model =
      CreateYolov8DetectionModel(infer_core, preprocess, postprocess, input_height, input_width,
                                 input_channels, cls_number, input_blobs_name, output_blobs_name);
  return yolov8_model;
}

static void benchmark_detection_2d_yolov8_rknn_sync(benchmark::State &state)
{
  benchmark_detection_2d_sync(state, CreateYolov8RknnModel());
}
static void benchmark_detection_2d_yolov8_rknn_async(benchmark::State &state)
{
  benchmark_detection_2d_async(state, CreateYolov8RknnModel());
}
BENCHMARK(benchmark_detection_2d_yolov8_rknn_sync)->Arg(500)->UseRealTime();
BENCHMARK(benchmark_detection_2d_yolov8_rknn_async)->Arg(500)->UseRealTime();

#endif

BENCHMARK_MAIN();
