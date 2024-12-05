/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-12-02 18:58:26
 * @LastEditTime: 2024-12-02 19:09:02
 * @FilePath: /easy_deploy/sam/sam_mobilesam/src/mobilesam_factory.cpp
 */
#include "sam_mobilesam/mobilesam.h"

namespace sam {

struct SamParams {
  std::shared_ptr<inference_core::BaseInferCoreFactory>         image_encoder_core_factory;
  std::shared_ptr<inference_core::BaseInferCoreFactory>         mask_points_decoder_core_factory;
  std::shared_ptr<inference_core::BaseInferCoreFactory>         mask_boxes_decoder_core_factory;
  std::shared_ptr<detection_2d::BaseDetectionPreprocessFactory> image_preprocess_block_factory;
  std::vector<std::string>                                      encoder_blob_names;
  std::vector<std::string>                                      box_dec_blob_names;
  std::vector<std::string>                                      point_dec_blob_names;
};

class SamMobileSamFactory : public BaseSamFactory {
public:
  SamMobileSamFactory(const SamParams &params) : params_(params)
  {}

  std::shared_ptr<sam::BaseSamModel> Create() override
  {
    return CreateMobileSamModel(params_.image_encoder_core_factory->Create(),
                                params_.mask_points_decoder_core_factory->Create(),
                                params_.mask_boxes_decoder_core_factory->Create(),
                                params_.image_preprocess_block_factory->Create(),
                                params_.encoder_blob_names, params_.box_dec_blob_names,
                                params_.point_dec_blob_names);
  }

private:
  SamParams params_;
};

std::shared_ptr<BaseSamFactory> CreateSamMobileSamModelFactory(
    std::shared_ptr<inference_core::BaseInferCoreFactory>         image_encoder_core_factory,
    std::shared_ptr<inference_core::BaseInferCoreFactory>         mask_points_decoder_core_factory,
    std::shared_ptr<inference_core::BaseInferCoreFactory>         mask_boxes_decoder_core_factory,
    std::shared_ptr<detection_2d::BaseDetectionPreprocessFactory> image_preprocess_block_factory,
    const std::vector<std::string>                               &encoder_blob_names,
    const std::vector<std::string>                               &box_dec_blob_names,
    const std::vector<std::string>                               &point_dec_blob_names)
{
  if (image_encoder_core_factory == nullptr || mask_points_decoder_core_factory == nullptr ||
      mask_boxes_decoder_core_factory == nullptr || image_preprocess_block_factory == nullptr)
  {
    throw std::invalid_argument("[CreateSamMobileSamModelFactory] Got invalid input arguments");
  }

  SamParams params;

  params.image_encoder_core_factory       = image_encoder_core_factory;
  params.mask_points_decoder_core_factory = mask_points_decoder_core_factory;
  params.mask_boxes_decoder_core_factory  = mask_boxes_decoder_core_factory;
  params.image_preprocess_block_factory   = image_preprocess_block_factory;
  params.encoder_blob_names               = encoder_blob_names;
  params.box_dec_blob_names               = box_dec_blob_names;
  params.point_dec_blob_names             = point_dec_blob_names;

  return std::make_shared<SamMobileSamFactory>(params);
}

} // namespace sam