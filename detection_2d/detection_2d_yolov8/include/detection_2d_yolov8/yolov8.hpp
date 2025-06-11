#pragma once

#include "deploy_core/base_detection.hpp"
#include "deploy_core/base_infer_core.hpp"

namespace easy_deploy {

/**
 * @brief Create a Yolov 8 Detection Model instance
 *
 * @param infer_core
 * @param preprocess_block
 * @param postprocess_block
 * @param input_height
 * @param input_width
 * @param input_channel
 * @param cls_number
 * @param input_blob_name
 * @param output_blob_name
 * @param downsample_scales
 * @return std::shared_ptr<BaseDetectionModel>
 */
std::shared_ptr<BaseDetectionModel> CreateYolov8DetectionModel(
    const std::shared_ptr<BaseInferCore>         &infer_core,
    const std::shared_ptr<IDetectionPreProcess>  &preprocess_block,
    const std::shared_ptr<IDetectionPostProcess> &postprocess_block,
    const int                                     input_height,
    const int                                     input_width,
    const int                                     input_channel,
    const int                                     cls_number,
    const std::vector<std::string>               &input_blob_name,
    const std::vector<std::string>               &output_blob_name,
    const std::vector<int>                        downsample_scales = {8, 16, 32});

std::shared_ptr<BaseDetection2DFactory> CreateYolov8DetectionModelFactory(
    std::shared_ptr<BaseInferCoreFactory>            infer_core_factory,
    std::shared_ptr<BaseDetectionPreprocessFactory>  preprocess_factory,
    std::shared_ptr<BaseDetectionPostprocessFactory> postprocess_factory,
    int                                              input_height,
    int                                              input_width,
    int                                              input_channel,
    int                                              cls_number,
    const std::vector<std::string>                  &input_blob_name,
    const std::vector<std::string>                  &output_blob_name,
    const std::vector<int>                          &downsample_scales = {8, 16, 32});

} // namespace easy_deploy
