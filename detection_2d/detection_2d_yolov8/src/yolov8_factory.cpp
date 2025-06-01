/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-12-02 19:08:10
 * @LastEditTime: 2024-12-02 19:31:43
 * @FilePath: /easy_deploy/detection_2d/detection_2d_yolov8/src/yolov8_factory.cpp
 */
#include "detection_2d_yolov8/yolov8.h"

namespace detection_2d {

struct Yolov8Params {
  std::shared_ptr<inference_core::BaseInferCoreFactory>          infer_core_factory;
  std::shared_ptr<detection_2d::BaseDetectionPreprocessFactory>  preprocess_factory;
  std::shared_ptr<detection_2d::BaseDetectionPostprocessFactory> postprocess_factory;
  int                                                            input_height;
  int                                                            input_width;
  int                                                            input_channel;
  int                                                            cls_number;
  std::vector<std::string>                                       input_blob_name;
  std::vector<std::string>                                       output_blob_name;
  std::vector<int>                                               downsample_scales;
};

class Detection2DYolov8Factory : public BaseDetection2DFactory {
public:
  Detection2DYolov8Factory(const Yolov8Params &params) : params_(params)
  {}
  std::shared_ptr<detection_2d::BaseDetectionModel> Create() override
  {
    return CreateYolov8DetectionModel(
        params_.infer_core_factory->Create(), params_.preprocess_factory->Create(),
        params_.postprocess_factory->Create(), params_.input_height, params_.input_width,
        params_.input_channel, params_.cls_number, params_.input_blob_name,
        params_.output_blob_name, params_.downsample_scales);
  }

private:
  Yolov8Params params_;
};

std::shared_ptr<BaseDetection2DFactory> CreateYolov8DetectionModelFactory(
    std::shared_ptr<inference_core::BaseInferCoreFactory>          infer_core_factory,
    std::shared_ptr<detection_2d::BaseDetectionPreprocessFactory>  preprocess_factory,
    std::shared_ptr<detection_2d::BaseDetectionPostprocessFactory> postprocess_factory,
    int                                                            input_height,
    int                                                            input_width,
    int                                                            input_channel,
    int                                                            cls_number,
    const std::vector<std::string>                                &input_blob_name,
    const std::vector<std::string>                                &output_blob_name,
    const std::vector<int>                                        &downsample_scales)
{
  if (infer_core_factory == nullptr || preprocess_factory == nullptr ||
      postprocess_factory == nullptr)
  {
    throw std::invalid_argument("[CreateYolov8DetectionModelFactory] Got invalid input arguments!");
  }

  Yolov8Params params;
  params.infer_core_factory  = infer_core_factory;
  params.preprocess_factory  = preprocess_factory;
  params.postprocess_factory = postprocess_factory;
  params.input_height        = input_height;
  params.input_width         = input_width;
  params.input_channel       = input_channel;
  params.cls_number          = cls_number;
  params.input_blob_name     = input_blob_name;
  params.output_blob_name    = output_blob_name;
  params.downsample_scales   = downsample_scales;

  return std::make_shared<Detection2DYolov8Factory>(params);
}

} // namespace detection_2d
