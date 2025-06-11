#include <gtest/gtest.h>

#include "detection_2d_util/detection_2d_util.hpp"
#include "detection_2d_yolov8/yolov8.hpp"
#include "test_utils/detection_2d_test_utils.hpp"

using namespace easy_deploy;

#define GEN_TEST_CASES(Tag, FixtureClass)                                                      \
  TEST_F(FixtureClass, test_yolov8_##Tag##_correctness)                                        \
  {                                                                                            \
    test_detection_2d_algorithm_correctness(yolov8_model_, test_image_path_, conf_threshold_,  \
                                            expected_obj_num_, test_visual_result_save_path_); \
  }                                                                                            \
  TEST_F(FixtureClass, test_yolov8_##Tag##_async_correctness)                                  \
  {                                                                                            \
    test_detection_2d_algorithm_async_correctness(yolov8_model_, test_image_path_,             \
                                                  conf_threshold_, expected_obj_num_,          \
                                                  test_visual_result_save_path_);              \
  }

class BaseYolov8Fixture : public testing::Test {
protected:
  std::shared_ptr<BaseDetectionModel> yolov8_model_;

  std::string test_image_path_;
  std::string test_visual_result_save_path_;
  float       conf_threshold_;
  size_t      expected_obj_num_;
};

#ifdef ENABLE_TENSORRT

#include "trt_core/trt_core.hpp"

class Yolov8_TensorRT_Fixture : public BaseYolov8Fixture {
public:
  void SetUp() override
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

    yolov8_model_ =
        CreateYolov8DetectionModel(infer_core, preprocess, postprocess, input_height, input_width,
                                   input_channels, cls_number, input_blobs_name, output_blobs_name);

    test_image_path_              = "/workspace/test_data/persons.jpg";
    test_visual_result_save_path_ = "/workspace/test_data/yolov8_tensorrt_test_result.jpg";
    conf_threshold_               = 0.4;
    expected_obj_num_             = 10ul;
  }
};

GEN_TEST_CASES(tensorrt, Yolov8_TensorRT_Fixture);

#endif

#ifdef ENABLE_ORT

#include "ort_core/ort_core.hpp"

class Yolov8_OnnxRuntime_Fixture : public BaseYolov8Fixture {
public:
  void SetUp() override
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

    yolov8_model_ =
        CreateYolov8DetectionModel(infer_core, preprocess, postprocess, input_height, input_width,
                                   input_channels, cls_number, input_blobs_name, output_blobs_name);

    test_image_path_              = "/workspace/test_data/persons.jpg";
    test_visual_result_save_path_ = "/workspace/test_data/yolov8_onnxruntime_test_result.jpg";
    conf_threshold_               = 0.4;
    expected_obj_num_             = 11ul;
  }
};

GEN_TEST_CASES(onnxruntime, Yolov8_OnnxRuntime_Fixture);

#endif

#ifdef ENABLE_RKNN

#include "rknn_core/rknn_core.hpp"

class Yolov8_Rknn_Fixture : public BaseYolov8Fixture {
public:
  void SetUp() override
  {
    std::string                    model_path     = "/workspace/models/yolov8n_divide_opset11.rknn";
    const int                      input_height   = 640;
    const int                      input_width    = 640;
    const int                      input_channels = 3;
    const int                      cls_number     = 80;
    const std::vector<std::string> input_blobs_name  = {"images"};
    const std::vector<std::string> output_blobs_name = {"318", "onnx::ReduceSum_326", "331",
                                                        "338", "onnx::ReduceSum_346", "350",
                                                        "357", "onnx::ReduceSum_365", "369"};

    auto infer_core  = CreateRknnInferCore(model_path, {{"images", RknnInputTensorType::RK_UINT8}});
    auto preprocess  = CreateCpuDetPreProcess({0, 0, 0}, {1, 1, 1}, false, false);
    auto postprocess = CreateYolov8PostProcessCpuDivide(input_height, input_width, cls_number);

    yolov8_model_ =
        CreateYolov8DetectionModel(infer_core, preprocess, postprocess, input_height, input_width,
                                   input_channels, cls_number, input_blobs_name, output_blobs_name);

    test_image_path_              = "/workspace/test_data/persons.jpg";
    test_visual_result_save_path_ = "/workspace/test_data/yolov8_rknn_test_result.jpg";
    conf_threshold_               = 0.4;
    expected_obj_num_             = 10ul;
  }
};

GEN_TEST_CASES(rknn, Yolov8_Rknn_Fixture);

#endif
