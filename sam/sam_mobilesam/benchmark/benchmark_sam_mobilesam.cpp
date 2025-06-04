#include <gtest/gtest.h>

#include "detection_2d_util/detection_2d_util.h"
#include "sam_mobilesam/mobilesam.h"
#include "benchmark_utils/sam_benchmark_utils.hpp"

using namespace inference_core;
using namespace detection_2d;
using namespace sam;
using namespace benchmark_utils;

#ifdef ENABLE_TENSORRT

#include "trt_core/trt_core.h"

std::shared_ptr<BaseSamModel> CreateSAMTensorRTModel(const std::string &image_encoder_model_path)
{
  auto box_decoder_model_path   = "/workspace/models/modified_mobile_sam_box.engine";
  auto point_decoder_model_path = "/workspace/models/modified_mobile_sam_point.engine";

  auto image_encoder = CreateTrtInferCore(image_encoder_model_path);

  const int SAM_MAX_BOX    = 1;
  const int SAM_MAX_POINTS = 8;

  auto box_decoder_factory =
      CreateTrtInferCoreFactory(box_decoder_model_path,
                                {
                                    {"image_embeddings", {1, 256, 64, 64}},
                                    {"boxes", {1, SAM_MAX_BOX, 4}},
                                    {"mask_input", {1, 1, 256, 256}},
                                    {"has_mask_input", {1}},
                                },
                                {{"masks", {1, 1, 256, 256}}, {"scores", {1, 1}}});

  auto point_decoder_factory =
      CreateTrtInferCoreFactory(point_decoder_model_path,
                                {
                                    {"image_embeddings", {1, 256, 64, 64}},
                                    {"point_coords", {1, SAM_MAX_POINTS, 2}},
                                    {"point_labels", {1, SAM_MAX_POINTS}},
                                    {"mask_input", {1, 1, 256, 256}},
                                    {"has_mask_input", {1}},
                                },
                                {{"masks", {1, 1, 256, 256}}, {"scores", {1, 1}}});

  auto image_preprocess_factory = CreateCudaDetPreProcessFactory();

  return CreateMobileSamModel(image_encoder, point_decoder_factory->Create(),
                              box_decoder_factory->Create(), image_preprocess_factory->Create());
}

// benchmark sam_mobilesam
static void benchmark_sam_mobilesam_tensorrt_sync(benchmark::State &state)
{
  auto mobilesam_image_encoder_model_path = "/workspace/models/mobile_sam_encoder.engine";
  benchmark_sam_sync(state, CreateSAMTensorRTModel(mobilesam_image_encoder_model_path));
}
static void benchmark_sam_mobilesam_tensorrt_async(benchmark::State &state)
{
  auto mobilesam_image_encoder_model_path = "/workspace/models/mobile_sam_encoder.engine";
  benchmark_sam_async(state, CreateSAMTensorRTModel(mobilesam_image_encoder_model_path));
}
BENCHMARK(benchmark_sam_mobilesam_tensorrt_sync)->Arg(100)->UseRealTime();
BENCHMARK(benchmark_sam_mobilesam_tensorrt_async)->Arg(100)->UseRealTime();

// benchmark sam_nanosam
static void benchmark_sam_nanosam_tensorrt_sync(benchmark::State &state)
{
  auto nanosam_image_encoder_model_path = "/workspace/models/nanosam_image_encoder_opset11.engine";
  benchmark_sam_sync(state, CreateSAMTensorRTModel(nanosam_image_encoder_model_path));
}
static void benchmark_sam_nanosam_tensorrt_async(benchmark::State &state)
{
  auto nanosam_image_encoder_model_path = "/workspace/models/nanosam_image_encoder_opset11.engine";
  benchmark_sam_async(state, CreateSAMTensorRTModel(nanosam_image_encoder_model_path));
}
BENCHMARK(benchmark_sam_nanosam_tensorrt_sync)->Arg(200)->UseRealTime();
BENCHMARK(benchmark_sam_nanosam_tensorrt_async)->Arg(200)->UseRealTime();

#endif

#ifdef ENABLE_ORT

#include "ort_core/ort_core.h"

std::shared_ptr<BaseSamModel> CreateSAMOnnxRuntimeModel(const std::string &image_encoder_model_path)
{
  auto box_decoder_model_path   = "/workspace/models/modified_mobile_sam_box.onnx";
  auto point_decoder_model_path = "/workspace/models/modified_mobile_sam_point.onnx";

  auto image_encoder = CreateOrtInferCore(image_encoder_model_path);

  const int SAM_MAX_BOX    = 1;
  const int SAM_MAX_POINTS = 8;

  auto box_decoder_factory =
      CreateOrtInferCoreFactory(box_decoder_model_path,
                                {
                                    {"image_embeddings", {1, 256, 64, 64}},
                                    {"boxes", {1, SAM_MAX_BOX, 4}},
                                    {"mask_input", {1, 1, 256, 256}},
                                    {"has_mask_input", {1}},
                                },
                                {{"masks", {1, 1, 256, 256}}, {"scores", {1, 1}}});

  auto point_decoder_factory =
      CreateOrtInferCoreFactory(point_decoder_model_path,
                                {
                                    {"image_embeddings", {1, 256, 64, 64}},
                                    {"point_coords", {1, SAM_MAX_POINTS, 2}},
                                    {"point_labels", {1, SAM_MAX_POINTS}},
                                    {"mask_input", {1, 1, 256, 256}},
                                    {"has_mask_input", {1}},
                                },
                                {{"masks", {1, 1, 256, 256}}, {"scores", {1, 1}}});

  auto image_preprocess_factory =
      CreateCpuDetPreProcessFactory({0, 0, 0}, {255, 255, 255}, true, true);

  return CreateMobileSamModel(image_encoder, point_decoder_factory->Create(),
                              box_decoder_factory->Create(), image_preprocess_factory->Create());
}

// benchmark sam_mobilesam
static void benchmark_sam_mobilesam_onnxruntime_sync(benchmark::State &state)
{
  auto mobilesam_image_encoder_model_path = "/workspace/models/mobile_sam_encoder.onnx";
  benchmark_sam_sync(state, CreateSAMOnnxRuntimeModel(mobilesam_image_encoder_model_path));
}
static void benchmark_sam_mobilesam_onnxruntime_async(benchmark::State &state)
{
  auto mobilesam_image_encoder_model_path = "/workspace/models/mobile_sam_encoder.onnx";
  benchmark_sam_async(state, CreateSAMOnnxRuntimeModel(mobilesam_image_encoder_model_path));
}
BENCHMARK(benchmark_sam_mobilesam_onnxruntime_sync)->Arg(20)->UseRealTime();
BENCHMARK(benchmark_sam_mobilesam_onnxruntime_async)->Arg(20)->UseRealTime();

// benchmark sam_nanosam
static void benchmark_sam_nanosam_onnxruntime_sync(benchmark::State &state)
{
  auto nanosam_image_encoder_model_path = "/workspace/models/nanosam_image_encoder_opset11.onnx";
  benchmark_sam_sync(state, CreateSAMOnnxRuntimeModel(nanosam_image_encoder_model_path));
}
static void benchmark_sam_nanosam_onnxruntime_async(benchmark::State &state)
{
  auto nanosam_image_encoder_model_path = "/workspace/models/nanosam_image_encoder_opset11.onnx";
  benchmark_sam_async(state, CreateSAMOnnxRuntimeModel(nanosam_image_encoder_model_path));
}
BENCHMARK(benchmark_sam_nanosam_onnxruntime_sync)->Arg(50)->UseRealTime();
BENCHMARK(benchmark_sam_nanosam_onnxruntime_async)->Arg(50)->UseRealTime();

#endif

#ifdef ENABLE_RKNN

#include "rknn_core/rknn_core.h"

std::shared_ptr<BaseSamModel> CreateSAMRknnModel(const std::string &image_encoder_model_path)
{
  auto box_decoder_model_path   = "/workspace/models/modified_mobile_sam_box.rknn";
  auto point_decoder_model_path = "/workspace/models/modified_mobile_sam_point.rknn";

  auto nanosam_image_encoder = CreateRknnInferCore(
      image_encoder_model_path, {{"images", RknnInputTensorType::RK_UINT8}}, 5, 2);

  auto box_decoder_factory = CreateRknnInferCoreFactory(box_decoder_model_path, {}, 5, 2);

  auto point_decoder_factory = CreateRknnInferCoreFactory(point_decoder_model_path, {}, 5, 2);

  auto image_preprocess_factory =
      CreateCpuDetPreProcessFactory({0, 0, 0}, {255, 255, 255}, false, false);

  return CreateMobileSamModel(nanosam_image_encoder, point_decoder_factory->Create(),
                              box_decoder_factory->Create(), image_preprocess_factory->Create());
}

// benchmark sam_nanosam
static void benchmark_sam_nanosam_rknn_sync(benchmark::State &state)
{
  auto nanosam_image_encoder_model_path = "/workspace/models/nanosam_image_encoder_opset11.rknn";
  benchmark_sam_sync(state, CreateSAMRknnModel(nanosam_image_encoder_model_path));
}
static void benchmark_sam_nanosam_rknn_async(benchmark::State &state)
{
  auto nanosam_image_encoder_model_path = "/workspace/models/nanosam_image_encoder_opset11.rknn";
  benchmark_sam_async(state, CreateSAMRknnModel(nanosam_image_encoder_model_path));
}
BENCHMARK(benchmark_sam_nanosam_rknn_sync)->Arg(50)->UseRealTime();
BENCHMARK(benchmark_sam_nanosam_rknn_async)->Arg(100)->UseRealTime();

#endif

BENCHMARK_MAIN();
