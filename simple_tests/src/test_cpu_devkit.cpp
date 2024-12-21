#include "tests/test_func.h"
#include "detection_2d_yolov8/yolov8.h"
#include "detection_2d_rt_detr/rt_detr.h"
#include "detection_2d_util/detection_2d_util.h"
#include "ort_core/ort_core.h"
#include "tests/fs_util.h"

#include "sam_mobilesam/mobilesam.h"

/**************************
****  ort core test ****
***************************/

using namespace inference_core;
using namespace detection_2d;
using namespace sam;

static
std::shared_ptr<BaseDetection2DFactory> GetYolov8Factory()
{
  std::string                    model_path        = "/workspace/models/yolov8n.onnx";
  const int                      input_height      = 640;
  const int                      input_width       = 640;
  const int                      input_channels    = 3;
  const int                      cls_number        = 80;
  const std::vector<std::string> input_blobs_name  = {"images"};
  const std::vector<std::string> output_blobs_name = {"output0"};

  auto infer_core_factory = CreateOrtInferCoreFactory(model_path);
  auto preprocess_factory = CreateCpuDetPreProcessFactory();
  auto postprocess_factory =
      CreateYolov8PostProcessCpuOriginFactory(input_height, input_width, cls_number);

  return CreateYolov8DetectionModelFactory(
      infer_core_factory, preprocess_factory, postprocess_factory, input_height, input_width,
      input_channels, cls_number, input_blobs_name, output_blobs_name);
}

static
std::shared_ptr<BaseDetection2DFactory> GetRTDetrFactory()
{
  std::string                    model_path     = "/workspace/models/rt_detr_v2_single_input.onnx";
  const int                      input_height   = 640;
  const int                      input_width    = 640;
  const int                      input_channels = 3;
  const int                      cls_number     = 80;
  const std::vector<std::string> input_blobs_name  = {"images"};
  const std::vector<std::string> output_blobs_name = {"labels", "boxes", "scores"};

  auto infer_core_factory = CreateOrtInferCoreFactory(model_path);
  auto preprocess_factory = CreateCpuDetPreProcessFactory();

  return CreateRTDetrDetectionModelFactory(infer_core_factory, preprocess_factory, input_height,
                                           input_width, input_channels, cls_number,
                                           input_blobs_name, output_blobs_name);
}

const static int SAM_MAX_POINTS = 8;
const static int SAM_MAX_BOX    = 1;

static
std::shared_ptr<BaseSamFactory> GetMobileSamFactory()
{
  auto image_encoder_model_path = "/workspace/models/mobile_sam_encoder.onnx";
  // auto image_encoder_model_path = "/workspace/models/nanosam_image_encoder_opset11.onnx";
  auto box_decoder_model_path   = "/workspace/models/modified_mobile_sam_box.onnx";
  auto point_decoder_model_path = "/workspace/models/modified_mobile_sam_point.onnx";

  auto image_encoder_factory = CreateOrtInferCoreFactory(image_encoder_model_path);

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

  auto image_preprocess_factory = CreateCpuDetPreProcessFactory();

  return CreateSamMobileSamModelFactory(image_encoder_factory, point_decoder_factory,
                                        box_decoder_factory, image_preprocess_factory);
}

TEST(detection_yolov8_test, ort_core_correctness)
{
  auto factory = GetYolov8Factory();
  int  res     = test_func_yolov8_model_correctness(factory->Create());
}

TEST(detection_yolov8_test, ort_core_speed)
{
  auto factory = GetYolov8Factory();
  int  res     = test_func_yolov8_model_speed(factory->Create());
}

TEST(detection_yolov8_test, ort_core_pipeline_correctness)
{
  auto factory = GetYolov8Factory();
  int  res     = test_func_yolov8_model_pipeline_correctness(factory->Create());
}

TEST(detection_yolov8_test, ort_core_pipeline_speed)
{
  auto factory = GetYolov8Factory();
  int  res     = test_func_yolov8_model_pipeline_speed(factory->Create());
}

TEST(detection_rtdetr_test, ort_core_correctness)
{
  auto factory = GetRTDetrFactory();
  int  res     = test_func_yolov8_model_correctness(factory->Create());
}

TEST(detection_rtdetr_test, ort_core_speed)
{
  auto factory = GetRTDetrFactory();
  int  res     = test_func_yolov8_model_speed(factory->Create());
}

TEST(detection_rtdetr_test, ort_core_pipeline_correctness)
{
  auto factory = GetRTDetrFactory();
  int  res     = test_func_yolov8_model_pipeline_correctness(factory->Create());
}

TEST(detection_rtdetr_test, ort_core_pipeline_speed)
{
  auto factory = GetRTDetrFactory();
  int  res     = test_func_yolov8_model_pipeline_speed(factory->Create());
}

TEST(sam_mobilesam_test, ort_with_point_correctness)
{
  auto factory = GetMobileSamFactory();
  test_func_sam_point_correctness(factory->Create());
}

TEST(sam_mobilesam_test, ort_with_point_speed)
{
  auto factory = GetMobileSamFactory();

  test_func_sam_point_speed(factory->Create());
}

TEST(sam_mobilesam_test, ort_with_box_correctness)
{
  auto factory = GetMobileSamFactory();

  test_func_sam_box_correctness(factory->Create());
}

TEST(sam_mobilesam_test, ort_with_point_pipeline_correctness)
{
  auto factory = GetMobileSamFactory();

  test_func_sam_point_pipeline_correctness(factory->Create());
}

TEST(sam_mobilesam_test, ort_with_point_pipeline_speed)
{
  auto factory = GetMobileSamFactory();

  test_func_sam_point_pipeline_speed(factory->Create());
}

TEST(sam_mobilesam_test, ort_with_box_pipeline_correctness)
{
  auto factory = GetMobileSamFactory();

  test_func_sam_box_pipeline_correctness(factory->Create());
}
