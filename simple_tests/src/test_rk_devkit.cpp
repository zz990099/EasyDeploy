#include "tests/test_func.h"
#include "rknn_core/rknn_core.h"
#include "detection_2d_yolov8/yolov8.h"
#include "detection_2d_util/detection_2d_util.h"
#include "sam_mobilesam/mobilesam.h"

/**************************
****  rknn core test ****
***************************/

using namespace inference_core;
using namespace detection_2d;
using namespace sam;

static
std::shared_ptr<BaseDetection2DFactory> GetYolov8Factory()
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

  auto infer_core_factory = CreateRknnInferCoreFactory(model_path, {}, 5, 3);
  auto preprocess_factory = CreateCpuDetPreProcessFactory({0, 0, 0}, {1, 1, 1}, false, false);
  auto postprocess_factory =
      CreateYolov8PostProcessCpuDivideFactory(input_height, input_width, cls_number);

  return CreateYolov8DetectionModelFactory(
      infer_core_factory, preprocess_factory, postprocess_factory, input_height, input_width,
      input_channels, cls_number, input_blobs_name, output_blobs_name);
}

static
std::shared_ptr<BaseSamFactory> GetMobileSamFactory()
{
  auto image_encoder_model_path = "/workspace/models/nanosam_image_encoder_opset11.rknn";
  auto box_decoder_model_path   = "/workspace/models/modified_mobile_sam_box.rknn";
  auto point_decoder_model_path = "/workspace/models/modified_mobile_sam_point.rknn";

  auto image_encoder_factory = CreateRknnInferCoreFactory(
      image_encoder_model_path, {{"images", RknnInputTensorType::RK_UINT8}}, 5, 2);

  auto box_decoder_factory =
      CreateRknnInferCoreFactory(box_decoder_model_path, {}, 5, 2);

  auto point_decoder_factory =
      CreateRknnInferCoreFactory(point_decoder_model_path, {}, 5, 2);

  auto image_preprocess_factory = CreateCpuDetPreProcessFactory({0, 0, 0}, {1, 1, 1}, false, false);

  return CreateSamMobileSamModelFactory(image_encoder_factory, point_decoder_factory,
                                        box_decoder_factory, image_preprocess_factory);
}

/***********************************
**** detection with rknn test ****
************************************/

TEST(detection_yolov8_test, rknn_core_correctness)
{
  auto factory = GetYolov8Factory();
  int  res     = test_func_yolov8_model_correctness(factory->Create());
}

TEST(detection_yolov8_test, rknn_core_speed)
{
  auto  factory = GetYolov8Factory();
  float fps     = test_func_yolov8_model_speed(factory->Create());
}

TEST(detection_yolov8_test, rknn_core_pipeline_correctness)
{
  auto factory = GetYolov8Factory();
  int  res     = test_func_yolov8_model_pipeline_correctness(factory->Create());
}

TEST(detection_yolov8_test, rknn_core_pipeline_speed)
{
  auto factory = GetYolov8Factory();
  int  res     = test_func_yolov8_model_pipeline_speed(factory->Create());
}

////////////////// 2024.10.18 UPDATED: SUPPORT all rknn mobilesam //////////////////
TEST(sam_mobilesam_test, rknn_with_point_all_rk_correctness)
{
  auto factory = GetMobileSamFactory();
  test_func_sam_point_correctness(factory->Create());
}

TEST(sam_mobilesam_test, rknn_with_box_all_rk_correctness)
{
  auto factory = GetMobileSamFactory();
  test_func_sam_box_correctness(factory->Create());
}

TEST(sam_mobilesam_test, rknn_with_point_all_rk_speed)
{
  auto factory = GetMobileSamFactory();
  test_func_sam_point_speed(factory->Create());
}

TEST(sam_mobilesam_test, rknn_with_box_all_rk_speed)
{
  auto factory = GetMobileSamFactory();
  test_func_sam_box_speed(factory->Create());
}

//////////////// pipeline ////////////////////

TEST(sam_mobilesam_test, rknn_with_point_all_rk_pipeline_correctness)
{
  auto factory = GetMobileSamFactory();
  test_func_sam_point_pipeline_correctness(factory->Create());
}

TEST(sam_mobilesam_test, rknn_with_box_all_rk_pipeline_correctness)
{
  auto factory = GetMobileSamFactory();
  test_func_sam_box_pipeline_correctness(factory->Create());
}

TEST(sam_mobilesam_test, rknn_with_point_all_rk_pipeline_speed)
{
  auto factory = GetMobileSamFactory();
  test_func_sam_point_pipeline_speed(factory->Create());
}

TEST(sam_mobilesam_test, rknn_with_box_all_rk_pipeline_speed)
{
  auto factory = GetMobileSamFactory();
  test_func_sam_box_pipeline_speed(factory->Create());
}