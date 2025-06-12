#include "eval_utils/detection_2d_eval_utils.hpp"
#include "detection_2d_util/detection_2d_util.hpp"
#include "detection_2d_yolov8/yolov8.hpp"

using namespace easy_deploy;

#ifdef ENABLE_TENSORRT

#include "trt_core/trt_core.hpp"

class EvalAccuracyYolov8TensorRTFixture : public EvalAccuracyDetection2DFixture {
public:
  SetUpReturnType SetUp() override
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

    const std::string coco_eval_dir_path = "/workspace/test_data/coco2017/coco2017_val";
    const std::string coco_annotations_path =
        "/workspace/test_data/coco2017/coco2017_annotations/instances_val2017.json";
    return {yolov8_model, coco_eval_dir_path, coco_annotations_path};
  }
};

RegisterEvalAccuracyDetection2D(EvalAccuracyYolov8TensorRTFixture);

#endif

#ifdef ENABLE_ORT

#include "ort_core/ort_core.hpp"

class EvalAccuracyYolov8OnnxRuntimeFixture : public EvalAccuracyDetection2DFixture {
public:
  SetUpReturnType SetUp() override
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

    const std::string coco_eval_dir_path = "/workspace/test_data/coco2017/coco2017_val";
    const std::string coco_annotations_path =
        "/workspace/test_data/coco2017/coco2017_annotations/instances_val2017.json";
    return {yolov8_model, coco_eval_dir_path, coco_annotations_path};
  }
};

RegisterEvalAccuracyDetection2D(EvalAccuracyYolov8OnnxRuntimeFixture);

#endif

#ifdef ENABLE_RKNN

#include "rknn_core/rknn_core.hpp"

class EvalAccuracyYolov8RknnFixture : public EvalAccuracyDetection2DFixture {
public:
  SetUpReturnType SetUp() override
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

    auto yolov8_model =
        CreateYolov8DetectionModel(infer_core, preprocess, postprocess, input_height, input_width,
                                   input_channels, cls_number, input_blobs_name, output_blobs_name);

    const std::string coco_eval_dir_path = "/workspace/test_data/coco2017/coco2017_val";
    const std::string coco_annotations_path =
        "/workspace/test_data/coco2017/coco2017_annotations/instances_val2017.json";
    return {yolov8_model, coco_eval_dir_path, coco_annotations_path};
  }
};

RegisterEvalAccuracyDetection2D(EvalAccuracyYolov8RknnFixture);

#endif

EVAL_MAIN()
