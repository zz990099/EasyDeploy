/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-26 19:41:34
 * @LastEditTime: 2024-12-02 18:55:08
 * @FilePath: /easy_deploy/sam/sam_mobilesam/include/sam_mobilesam/mobilesam.h
 */
#ifndef __EASY_DEPLOY_MOBILE_SAM_H
#define __EASY_DEPLOY_MOBILE_SAM_H

#include "deploy_core/base_sam.h"
#include "deploy_core/base_detection.h"

namespace sam {

std::shared_ptr<BaseSamModel> CreateMobileSamModel(
    std::shared_ptr<inference_core::BaseInferCore>      image_encoder_core,
    std::shared_ptr<inference_core::BaseInferCore>      mask_points_decoder_core,
    std::shared_ptr<inference_core::BaseInferCore>      mask_boxes_decoder_core,
    std::shared_ptr<detection_2d::IDetectionPreProcess> image_preprocess_block,
    const std::vector<std::string>                     &encoder_blob_names = {"images", "features"},
    const std::vector<std::string> &box_dec_blob_names = {"image_embeddings", "boxes", "mask_input",
                                                          "has_mask_input", "masks", "scores"},
    const std::vector<std::string> &point_dec_blob_names = {"image_embeddings", "point_coords",
                                                            "point_labels", "mask_input",
                                                            "has_mask_input", "masks", "scores"});

std::shared_ptr<BaseSamFactory> CreateSamMobileSamModelFactory(
    std::shared_ptr<inference_core::BaseInferCoreFactory>         image_encoder_core_factory,
    std::shared_ptr<inference_core::BaseInferCoreFactory>         mask_points_decoder_core_factory,
    std::shared_ptr<inference_core::BaseInferCoreFactory>         mask_boxes_decoder_core_factory,
    std::shared_ptr<detection_2d::BaseDetectionPreprocessFactory> image_preprocess_block_factory,
    const std::vector<std::string> &encoder_blob_names = {"images", "features"},
    const std::vector<std::string> &box_dec_blob_names = {"image_embeddings", "boxes", "mask_input",
                                                          "has_mask_input", "masks", "scores"},
    const std::vector<std::string> &point_dec_blob_names = {"image_embeddings", "point_coords",
                                                            "point_labels", "mask_input",
                                                            "has_mask_input", "masks", "scores"});

} // namespace sam
#endif
