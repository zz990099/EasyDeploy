#pragma once

#include "deploy_core/base_sam.hpp"
#include "deploy_core/base_detection.hpp"

namespace easy_deploy {

std::shared_ptr<BaseSamModel> CreateMobileSamModel(
    std::shared_ptr<BaseInferCore>        image_encoder_core,
    std::shared_ptr<BaseInferCore>        mask_points_decoder_core,
    std::shared_ptr<BaseInferCore>        mask_boxes_decoder_core,
    std::shared_ptr<IDetectionPreProcess> image_preprocess_block,
    const std::vector<std::string>       &encoder_blob_names = {"images", "features"},
    const std::vector<std::string> &box_dec_blob_names = {"image_embeddings", "boxes", "mask_input",
                                                          "has_mask_input", "masks", "scores"},
    const std::vector<std::string> &point_dec_blob_names = {"image_embeddings", "point_coords",
                                                            "point_labels", "mask_input",
                                                            "has_mask_input", "masks", "scores"});

std::shared_ptr<BaseSamFactory> CreateSamMobileSamModelFactory(
    std::shared_ptr<BaseInferCoreFactory>           image_encoder_core_factory,
    std::shared_ptr<BaseInferCoreFactory>           mask_points_decoder_core_factory,
    std::shared_ptr<BaseInferCoreFactory>           mask_boxes_decoder_core_factory,
    std::shared_ptr<BaseDetectionPreprocessFactory> image_preprocess_block_factory,
    const std::vector<std::string>                 &encoder_blob_names = {"images", "features"},
    const std::vector<std::string> &box_dec_blob_names = {"image_embeddings", "boxes", "mask_input",
                                                          "has_mask_input", "masks", "scores"},
    const std::vector<std::string> &point_dec_blob_names = {"image_embeddings", "point_coords",
                                                            "point_labels", "mask_input",
                                                            "has_mask_input", "masks", "scores"});

} // namespace easy_deploy
