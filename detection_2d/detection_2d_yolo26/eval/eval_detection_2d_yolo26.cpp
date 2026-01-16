#include "eval_utils/detection_2d_eval_utils.hpp"
#include "detection_2d_util/detection_2d_util.hpp"
#include "detection_2d_yolo26/yolo26.hpp"

using namespace easy_deploy;

#ifdef ENABLE_TENSORRT

#include "trt_core/trt_core.hpp"

class EvalAccuracyYolo26TensorRTFixture : public EvalAccuracyDetection2DFixture {
public:
  SetUpReturnType SetUp() override
  {
    std::string                    model_path = "/workspace/models/yolo26n.engine";
    const int                      input_height      = 640;
    const int                      input_width       = 640;
    const int                      input_channels    = 3;
    const int                      cls_number        = 80;
    const std::vector<std::string> input_blobs_name  = {"images"};
    const std::vector<std::string> output_blobs_name = {"output0"};

    auto infer_core = CreateTrtInferCore(model_path);
    auto preprocess = CreateCpuDetPreProcess({0, 0, 0}, {255, 255, 255}, true, true);

    auto yolo26_model =
        CreateYolo26DetectionModel(infer_core, preprocess, input_height, input_width,
                                   input_channels, cls_number, input_blobs_name, output_blobs_name);
    const std::string coco_eval_dir_path = "/workspace/test_data/coco2017/coco2017_val";
    const std::string coco_annotations_path =
        "/workspace/test_data/coco2017/coco2017_annotations/instances_val2017.json";
    return {yolo26_model, coco_eval_dir_path, coco_annotations_path};
  }
};

RegisterEvalAccuracyDetection2D(EvalAccuracyYolo26TensorRTFixture);

#endif

#ifdef ENABLE_ORT

#include "ort_core/ort_core.hpp"

class EvalAccuracyYolo26OnnxRuntimeFixture : public EvalAccuracyDetection2DFixture {
public:
  SetUpReturnType SetUp() override
  {
    std::string                    model_path   = "/workspace/models/yolo26n.onnx";
    const int                      input_height = 640;
    const int                      input_width  = 640;
    const int                      input_channels    = 3;
    const int                      cls_number        = 80;
    const std::vector<std::string> input_blobs_name  = {"images"};
    const std::vector<std::string> output_blobs_name = {"output0"};

    auto infer_core = CreateOrtInferCore(model_path);
    auto preprocess = CreateCpuDetPreProcess({0, 0, 0}, {255, 255, 255}, true, true);

    auto yolo26_model =
        CreateYolo26DetectionModel(infer_core, preprocess, input_height, input_width,
                                   input_channels, cls_number, input_blobs_name, output_blobs_name);

    const std::string coco_eval_dir_path = "/workspace/test_data/coco2017/coco2017_val";
    const std::string coco_annotations_path =
        "/workspace/test_data/coco2017/coco2017_annotations/instances_val2017.json";
    return {yolo26_model, coco_eval_dir_path, coco_annotations_path};
  }
};

RegisterEvalAccuracyDetection2D(EvalAccuracyYolo26OnnxRuntimeFixture);

#endif

EVAL_MAIN()
