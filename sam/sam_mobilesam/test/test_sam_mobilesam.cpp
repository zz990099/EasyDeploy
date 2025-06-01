#include <gtest/gtest.h>

#include "detection_2d_util/detection_2d_util.h"
#include "sam_mobilesam/mobilesam.h"
#include "test_utils/sam_test_utils.hpp"

using namespace inference_core;
using namespace detection_2d;
using namespace sam;
using namespace test_utils;

#define GEN_MOBILESAM_TEST_CASES(Tag, FixtureClass)                                             \
  TEST_F(FixtureClass, test_mobilesam_##Tag##_correctness_with_points)                          \
  {                                                                                             \
    test_sam_algorithm_correctness_with_points(mobilesam_model_, points_, labels_,              \
                                               test_image_path_,                                \
                                               test_mobilesam_visual_result_save_path_);        \
  }                                                                                             \
  TEST_F(FixtureClass, test_mobilesam_##Tag##_async_correctness_with_points)                    \
  {                                                                                             \
    test_sam_algorithm_async_correctness_with_points(mobilesam_model_, points_, labels_,        \
                                                     test_image_path_,                          \
                                                     test_mobilesam_visual_result_save_path_);  \
  }                                                                                             \
  TEST_F(FixtureClass, test_mobilesam_##Tag##_correctness_with_boxes)                           \
  {                                                                                             \
    test_sam_algorithm_correctness_with_boxes(mobilesam_model_, boxes_, test_image_path_,       \
                                              test_mobilesam_visual_result_save_path_);         \
  }                                                                                             \
  TEST_F(FixtureClass, test_mobilesam_##Tag##_async_correctness_with_boxes)                     \
  {                                                                                             \
    test_sam_algorithm_async_correctness_with_boxes(mobilesam_model_, boxes_, test_image_path_, \
                                                    test_mobilesam_visual_result_save_path_);   \
  }

#define GEN_NANOSAM_TEST_CASES(Tag, FixtureClass)                                                  \
  TEST_F(FixtureClass, test_nanosam_##Tag##_correctness_with_points)                               \
  {                                                                                                \
    test_sam_algorithm_correctness_with_points(nanosam_model_, points_, labels_, test_image_path_, \
                                               test_nanosam_visual_result_save_path_);             \
  }                                                                                                \
  TEST_F(FixtureClass, test_nanosam_##Tag##_async_correctness_with_points)                         \
  {                                                                                                \
    test_sam_algorithm_async_correctness_with_points(nanosam_model_, points_, labels_,             \
                                                     test_image_path_,                             \
                                                     test_nanosam_visual_result_save_path_);       \
  }                                                                                                \
  TEST_F(FixtureClass, test_nanosam_##Tag##_correctness_with_boxes)                                \
  {                                                                                                \
    test_sam_algorithm_correctness_with_boxes(nanosam_model_, boxes_, test_image_path_,            \
                                              test_nanosam_visual_result_save_path_);              \
  }                                                                                                \
  TEST_F(FixtureClass, test_nanosam_##Tag##_async_correctness_with_boxes)                          \
  {                                                                                                \
    test_sam_algorithm_async_correctness_with_boxes(nanosam_model_, boxes_, test_image_path_,      \
                                                    test_nanosam_visual_result_save_path_);        \
  }

class BaseSamFixture : public testing::Test {
protected:
  std::shared_ptr<BaseSamModel> mobilesam_model_;
  std::shared_ptr<BaseSamModel> nanosam_model_;

  std::string test_image_path_;
  std::string test_mobilesam_visual_result_save_path_;
  std::string test_nanosam_visual_result_save_path_;

  std::vector<std::pair<int, int>> points_;
  std::vector<int>                 labels_;
  std::vector<BBox2D>              boxes_;
};

#ifdef ENABLE_TENSORRT

#include "trt_core/trt_core.h"

class Sam_TensorRT_Fixture : public BaseSamFixture {
public:
  void SetUp() override
  {
    auto mobilesam_image_encoder_model_path = "/workspace/models/mobile_sam_encoder.engine";
    auto nanosam_image_encoder_model_path =
        "/workspace/models/nanosam_image_encoder_opset11.engine";
    auto box_decoder_model_path   = "/workspace/models/modified_mobile_sam_box.engine";
    auto point_decoder_model_path = "/workspace/models/modified_mobile_sam_point.engine";

    auto mobilesam_image_encoder = CreateTrtInferCore(mobilesam_image_encoder_model_path);
    auto nanosam_image_encoder   = CreateTrtInferCore(mobilesam_image_encoder_model_path);

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

    mobilesam_model_ =
        CreateMobileSamModel(mobilesam_image_encoder, point_decoder_factory->Create(),
                             box_decoder_factory->Create(), image_preprocess_factory->Create());

    nanosam_model_ =
        CreateMobileSamModel(nanosam_image_encoder, point_decoder_factory->Create(),
                             box_decoder_factory->Create(), image_preprocess_factory->Create());

    test_image_path_ = "/workspace/test_data/persons.jpg";
    test_mobilesam_visual_result_save_path_ =
        "/workspace/test_data/mobilesam_tensorrt_test_result.jpg";
    test_nanosam_visual_result_save_path_ = "/workspace/test_data/nanosam_tensorrt_test_result.jpg";

    points_ = {{225, 370}};
    labels_ = {1};

    BBox2D box;
    box.x  = 225;
    box.y  = 370;
    box.w  = 110;
    box.h  = 300;
    boxes_ = {box};
  }
};

GEN_MOBILESAM_TEST_CASES(tensorrt, Sam_TensorRT_Fixture);
GEN_NANOSAM_TEST_CASES(tensorrt, Sam_TensorRT_Fixture);

#endif

#ifdef ENABLE_ORT

#include "ort_core/ort_core.h"

class Sam_OnnxRuntime_Fixture : public BaseSamFixture {
public:
  void SetUp() override
  {
    auto mobilesam_image_encoder_model_path = "/workspace/models/mobile_sam_encoder.onnx";
    auto nanosam_image_encoder_model_path = "/workspace/models/nanosam_image_encoder_opset11.onnx";
    auto box_decoder_model_path           = "/workspace/models/modified_mobile_sam_box.onnx";
    auto point_decoder_model_path         = "/workspace/models/modified_mobile_sam_point.onnx";

    auto mobilesam_image_encoder = CreateOrtInferCore(mobilesam_image_encoder_model_path);
    auto nanosam_image_encoder   = CreateOrtInferCore(mobilesam_image_encoder_model_path);

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

    auto image_preprocess_factory = CreateCpuDetPreProcessFactory({0, 0, 0}, {255, 255, 255}, true, true);

    mobilesam_model_ =
        CreateMobileSamModel(mobilesam_image_encoder, point_decoder_factory->Create(),
                             box_decoder_factory->Create(), image_preprocess_factory->Create());

    nanosam_model_ =
        CreateMobileSamModel(nanosam_image_encoder, point_decoder_factory->Create(),
                             box_decoder_factory->Create(), image_preprocess_factory->Create());

    test_image_path_ = "/workspace/test_data/persons.jpg";
    test_mobilesam_visual_result_save_path_ =
        "/workspace/test_data/mobilesam_onnxruntime_test_result.jpg";
    test_nanosam_visual_result_save_path_ =
        "/workspace/test_data/nanosam_onnxruntime_test_result.jpg";

    points_ = {{225, 370}};
    labels_ = {1};

    BBox2D box;
    box.x  = 225;
    box.y  = 370;
    box.w  = 110;
    box.h  = 300;
    boxes_ = {box};
  }
};

GEN_MOBILESAM_TEST_CASES(onnxruntime, Sam_OnnxRuntime_Fixture);
GEN_NANOSAM_TEST_CASES(onnxruntime, Sam_OnnxRuntime_Fixture);

#endif
