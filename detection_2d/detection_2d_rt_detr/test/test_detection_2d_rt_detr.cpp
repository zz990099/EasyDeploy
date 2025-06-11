#include <gtest/gtest.h>

#include "detection_2d_util/detection_2d_util.hpp"
#include "detection_2d_rt_detr/rt_detr.hpp"
#include "test_utils/detection_2d_test_utils.hpp"

using namespace easy_deploy;

#define GEN_TEST_CASES(Tag, FixtureClass)                                                      \
  TEST_F(FixtureClass, test_rt_detr_##Tag##_correctness)                                       \
  {                                                                                            \
    test_detection_2d_algorithm_correctness(rt_detr_model_, test_image_path_, conf_threshold_, \
                                            expected_obj_num_, test_visual_result_save_path_); \
  }                                                                                            \
  TEST_F(FixtureClass, test_rt_detr_##Tag##_async_correctness)                                 \
  {                                                                                            \
    test_detection_2d_algorithm_async_correctness(rt_detr_model_, test_image_path_,            \
                                                  conf_threshold_, expected_obj_num_,          \
                                                  test_visual_result_save_path_);              \
  }

class BaseRTDetrFixture : public testing::Test {
protected:
  std::shared_ptr<BaseDetectionModel> rt_detr_model_;

  std::string test_image_path_;
  std::string test_visual_result_save_path_;
  float       conf_threshold_;
  size_t      expected_obj_num_;
};

#ifdef ENABLE_TENSORRT

#include "trt_core/trt_core.hpp"

class RTDetr_TensorRT_Fixture : public BaseRTDetrFixture {
public:
  void SetUp() override
  {
    std::string                    model_path = "/workspace/models/rt_detr_v2_single_input.engine";
    const int                      input_height      = 640;
    const int                      input_width       = 640;
    const int                      input_channels    = 3;
    const int                      cls_number        = 80;
    const std::vector<std::string> input_blobs_name  = {"images"};
    const std::vector<std::string> output_blobs_name = {"labels", "boxes", "scores"};

    auto infer_core = CreateTrtInferCore(model_path);
    auto preprocess = CreateCudaDetPreProcess();

    rt_detr_model_ =
        CreateRTDetrDetectionModel(infer_core, preprocess, input_height, input_width,
                                   input_channels, cls_number, input_blobs_name, output_blobs_name);

    test_image_path_              = "/workspace/test_data/persons.jpg";
    test_visual_result_save_path_ = "/workspace/test_data/rt_detr_tensorrt_test_result.jpg";
    conf_threshold_               = 0.4;
    expected_obj_num_             = 22ul;
  }
};

GEN_TEST_CASES(tensorrt, RTDetr_TensorRT_Fixture);

#endif

#ifdef ENABLE_ORT

#include "ort_core/ort_core.hpp"

class RTDetr_OnnxRuntime_Fixture : public BaseRTDetrFixture {
public:
  void SetUp() override
  {
    std::string                    model_path   = "/workspace/models/rt_detr_v2_single_input.onnx";
    const int                      input_height = 640;
    const int                      input_width  = 640;
    const int                      input_channels    = 3;
    const int                      cls_number        = 80;
    const std::vector<std::string> input_blobs_name  = {"images"};
    const std::vector<std::string> output_blobs_name = {"labels", "boxes", "scores"};

    auto infer_core = CreateOrtInferCore(model_path);
    auto preprocess = CreateCpuDetPreProcess({0, 0, 0}, {255, 255, 255}, true, true);

    rt_detr_model_ =
        CreateRTDetrDetectionModel(infer_core, preprocess, input_height, input_width,
                                   input_channels, cls_number, input_blobs_name, output_blobs_name);

    test_image_path_              = "/workspace/test_data/persons.jpg";
    test_visual_result_save_path_ = "/workspace/test_data/rt_detr_onnxruntime_test_result.jpg";
    conf_threshold_               = 0.4;
    expected_obj_num_             = 23ul;
  }
};

GEN_TEST_CASES(onnxruntime, RTDetr_OnnxRuntime_Fixture);

#endif
