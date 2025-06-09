#include "eval_utils/detection_2d_eval_utils.hpp"
#include "detection_2d_util/detection_2d_util.h"
#include "detection_2d_rt_detr/rt_detr.h"

using namespace inference_core;
using namespace detection_2d;
using namespace eval_utils;

#ifdef ENABLE_TENSORRT

#include "trt_core/trt_core.h"

class EvalAccuracyRTDetrTensorRTFixture : public EvalAccuracyDetection2DFixture {
public:
  SetUpReturnType SetUp() override
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

    auto rt_detr_model =
        CreateRTDetrDetectionModel(infer_core, preprocess, input_height, input_width,
                                   input_channels, cls_number, input_blobs_name, output_blobs_name);
    const std::string coco_eval_dir_path = "/workspace/test_data/coco2017/coco2017_val";
    const std::string coco_annotations_path =
        "/workspace/test_data/coco2017/coco2017_annotations/instances_val2017.json";
    return {rt_detr_model, coco_eval_dir_path, coco_annotations_path};
  }
};

RegisterEvalAccuracyDetection2D(EvalAccuracyRTDetrTensorRTFixture);

#endif

#ifdef ENABLE_ORT

#include "ort_core/ort_core.h"

class EvalAccuracyRTDetrOnnxRuntimeFixture : public EvalAccuracyDetection2DFixture {
public:
  SetUpReturnType SetUp() override
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

    auto rt_detr_model =
        CreateRTDetrDetectionModel(infer_core, preprocess, input_height, input_width,
                                   input_channels, cls_number, input_blobs_name, output_blobs_name);

    const std::string coco_eval_dir_path = "/workspace/test_data/coco2017/coco2017_val";
    const std::string coco_annotations_path =
        "/workspace/test_data/coco2017/coco2017_annotations/instances_val2017.json";
    return {rt_detr_model, coco_eval_dir_path, coco_annotations_path};
  }
};

RegisterEvalAccuracyDetection2D(EvalAccuracyRTDetrOnnxRuntimeFixture);

#endif

EVAL_MAIN()