#pragma once

#include "deploy_core/base_detection.hpp"

namespace easy_deploy {

std::shared_ptr<BaseDetectionModel> CreateRTDetrDetectionModel(
    const std::shared_ptr<BaseInferCore>        &infer_core,
    const std::shared_ptr<IDetectionPreProcess> &preprocess_block,
    const int                                    input_height,
    const int                                    input_width,
    const int                                    input_channel,
    const int                                    cls_number,
    const std::vector<std::string>              &input_blobs_name  = {"images"},
    const std::vector<std::string>              &output_blobs_name = {"labels", "boxes", "scores"});

std::shared_ptr<BaseDetection2DFactory> CreateRTDetrDetectionModelFactory(
    std::shared_ptr<BaseInferCoreFactory>           infer_core_factory,
    std::shared_ptr<BaseDetectionPreprocessFactory> preprocess_factory,
    int                                             input_height,
    int                                             input_width,
    int                                             input_channel,
    int                                             cls_number,
    const std::vector<std::string>                 &input_blob_name,
    const std::vector<std::string>                 &output_blob_name);

} // namespace easy_deploy
