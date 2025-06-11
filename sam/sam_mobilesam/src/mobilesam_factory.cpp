#include "sam_mobilesam/mobilesam.hpp"

namespace easy_deploy {

struct SamParams {
  std::shared_ptr<BaseInferCoreFactory>           image_encoder_core_factory;
  std::shared_ptr<BaseInferCoreFactory>           mask_points_decoder_core_factory;
  std::shared_ptr<BaseInferCoreFactory>           mask_boxes_decoder_core_factory;
  std::shared_ptr<BaseDetectionPreprocessFactory> image_preprocess_block_factory;
  std::vector<std::string>                        encoder_blob_names;
  std::vector<std::string>                        box_dec_blob_names;
  std::vector<std::string>                        point_dec_blob_names;
};

class SamMobileSamFactory : public BaseSamFactory {
public:
  SamMobileSamFactory(const SamParams &params) : params_(params)
  {}

  std::shared_ptr<BaseSamModel> Create() override
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
    std::shared_ptr<BaseInferCoreFactory>           image_encoder_core_factory,
    std::shared_ptr<BaseInferCoreFactory>           mask_points_decoder_core_factory,
    std::shared_ptr<BaseInferCoreFactory>           mask_boxes_decoder_core_factory,
    std::shared_ptr<BaseDetectionPreprocessFactory> image_preprocess_block_factory,
    const std::vector<std::string>                 &encoder_blob_names,
    const std::vector<std::string>                 &box_dec_blob_names,
    const std::vector<std::string>                 &point_dec_blob_names)
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

} // namespace easy_deploy
