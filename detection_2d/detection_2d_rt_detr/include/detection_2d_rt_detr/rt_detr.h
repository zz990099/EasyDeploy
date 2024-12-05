/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-26 19:41:34
 * @LastEditTime: 2024-12-02 19:22:30
 * @FilePath: /easy_deploy/detection_2d/detection_2d_rt_detr/include/detection_2d_rt_detr/rt_detr.h
 */
#ifndef __DETECTION_RTDETR_H
#define __DETECTION_RTDETR_H

#include "deploy_core/base_detection.h"

namespace detection_2d {

std::shared_ptr<BaseDetectionModel> CreateRTDetrDetectionModel(
    const std::shared_ptr<inference_core::BaseInferCore> &infer_core,
    const std::shared_ptr<IDetectionPreProcess>          &preprocess_block,
    const int                                             input_height,
    const int                                             input_width,
    const int                                             input_channel,
    const int                                             cls_number,
    const std::vector<std::string>                       &input_blobs_name = {"images"},
    const std::vector<std::string> &output_blobs_name = {"labels", "boxes", "scores"});

std::shared_ptr<BaseDetection2DFactory> CreateRTDetrDetectionModelFactory(
    std::shared_ptr<inference_core::BaseInferCoreFactory>         infer_core_factory,
    std::shared_ptr<detection_2d::BaseDetectionPreprocessFactory> preprocess_factory,
    int                                                           input_height,
    int                                                           input_width,
    int                                                           input_channel,
    int                                                           cls_number,
    const std::vector<std::string>                               &input_blob_name,
    const std::vector<std::string>                               &output_blob_name);

} // namespace detection_2d
#endif