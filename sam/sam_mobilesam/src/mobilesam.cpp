#include "sam_mobilesam/mobilesam.h"

#include "deploy_core/wrapper.h"

namespace sam {

static void ThrowRuntimeError(const std::string &hint, uint64_t line_num)
{
  std::string exception_message = "[MobileSam:" + std::to_string(line_num) + "] " + hint;
  throw std::runtime_error(exception_message);
}

static void CheckBlobNameMatched(const std::string &infer_core_name,
                                 const std::shared_ptr<inference_core::BaseInferCore> &infer_core,
                                 const std::vector<std::string>                       &blob_names)
{
  auto blob_buffer = infer_core->AllocBlobsBuffer();
  if (blob_names.size() != blob_buffer->Size())
  {
    ThrowRuntimeError(infer_core_name + " core got different blob size with blob_names input!" +
                          std::to_string(blob_buffer->Size()) + " vs " +
                          std::to_string(blob_names.size()),
                      __LINE__);
  }
  for (const auto &blob_name : blob_names)
  {
    try
    {
      auto buffer_ptr = blob_buffer->GetOuterBlobBuffer(blob_name);
    } catch (std::exception e)
    {
      ThrowRuntimeError(infer_core_name + " met invalid blob_name in blob_names : " + blob_name,
                        __LINE__);
    }
  }
}

static void rknn_nchw_2_nhwc(float *nchw, float *nhwc, int N, int C, int H, int W)
{
  for (int ni = 0; ni < N; ni++)
  {
    for (int hi = 0; hi < H; hi++)
    {
      for (int wi = 0; wi < W; wi++)
      {
        for (int ci = 0; ci < C; ci++)
        {
          memcpy(nhwc + ni * H * W * C + hi * W * C + wi * C + ci,
                 nchw + ni * C * H * W + ci * H * W + hi * W + wi, sizeof(float));
        }
      }
    }
  }
}

class MobileSam : public BaseSamModel {
public:
  MobileSam(std::shared_ptr<inference_core::BaseInferCore>      image_encoder_core,
            std::shared_ptr<inference_core::BaseInferCore>      mask_points_decoder_core,
            std::shared_ptr<inference_core::BaseInferCore>      mask_boxes_decoder_core,
            std::shared_ptr<detection_2d::IDetectionPreProcess> image_preprocess_block,
            const std::vector<std::string>                     &encoder_blob_names,
            const std::vector<std::string>                     &box_dec_blob_names,
            const std::vector<std::string>                     &point_dec_blob_names);

  ~MobileSam() = default;

private:
  bool ImagePreProcess(ParsingType pipeline_unit) override;

  bool PromptBoxPreProcess(ParsingType pipeline_unit) override;

  bool PromptPointPreProcess(ParsingType pipeline_unit) override;

  bool MaskPostProcess(ParsingType pipeline_unit) override;

public:
  static const std::string model_name_;

private:
  const std::vector<std::string> encoder_blob_names_;
  const std::vector<std::string> box_dec_blob_names_;
  const std::vector<std::string> point_dec_blob_names_;

  std::shared_ptr<detection_2d::IDetectionPreProcess> image_preprocess_block_;

private:
  // defualt params, no access provided to user
  const int         IMAGE_INPUT_HEIGHT   = 1024;
  const int         IMAGE_INPUT_WIDTH    = 1024;
  const int         IMAGE_FEATURE_HEIGHT = 64;
  const int         IMAGE_FEATURE_WIDTH  = 64;
  const int         IMAGE_FEATURES_LEN   = 256;
  const int         MASK_LOW_RES_HEIGHT  = 256;
  const int         MASK_LOW_RES_WIDTH   = 256;
  const std::string MASK_OUT_BLOB_NAME   = "masks";
};

const std::string MobileSam::model_name_ = "MobileSam";


MobileSam::MobileSam(std::shared_ptr<inference_core::BaseInferCore>      image_encoder_core,
                     std::shared_ptr<inference_core::BaseInferCore>      mask_points_decoder_core,
                     std::shared_ptr<inference_core::BaseInferCore>      mask_boxes_decoder_core,
                     std::shared_ptr<detection_2d::IDetectionPreProcess> image_preprocess_block,
                     const std::vector<std::string>                     &encoder_blob_names,
                     const std::vector<std::string>                     &box_dec_blob_names,
                     const std::vector<std::string>                     &point_dec_blob_names)
    : BaseSamModel(
          model_name_, image_encoder_core, mask_points_decoder_core, mask_boxes_decoder_core),
      image_preprocess_block_(image_preprocess_block),
      encoder_blob_names_(encoder_blob_names),
      box_dec_blob_names_(box_dec_blob_names),
      point_dec_blob_names_(point_dec_blob_names)
{
  // Check
  CheckBlobNameMatched("image_encoder", image_encoder_core, encoder_blob_names);
  if (mask_boxes_decoder_core != nullptr)
    CheckBlobNameMatched("box_decoder", mask_boxes_decoder_core, box_dec_blob_names);
  if (mask_points_decoder_core != nullptr)
    CheckBlobNameMatched("point_decoder", mask_points_decoder_core, point_dec_blob_names);

  if (image_preprocess_block_ == nullptr)
  {
    throw std::invalid_argument("[MobileSAM] Got INVALID preprocess_block ptr!!!");
  }
}

bool MobileSam::ImagePreProcess(ParsingType package)
{
  auto p_package = std::dynamic_pointer_cast<SamPipelinePackage>(package);
  CHECK_STATE(p_package != nullptr,
              "[MobileSam Image PreProcess] the `package` instance \
                                    is not a instance of `SamPipelinePackage`!");

  auto encoder_blobs_buffer = p_package->image_encoder_blobs_buffer;
  // make the output buffer at device side 
  // (some inference framework will still output buffer to host side)
  p_package->image_encoder_blobs_buffer->SetBlobBuffer(encoder_blob_names_[1],
                                                       DataLocation::DEVICE);

  // preprocess image and write into buffer
  const auto scale = image_preprocess_block_->Preprocess(
      p_package->input_image_data, encoder_blobs_buffer, encoder_blob_names_[0], IMAGE_INPUT_HEIGHT,
      IMAGE_INPUT_WIDTH);
  // set the inference buffer
  p_package->infer_buffer = p_package->image_encoder_blobs_buffer;
  // record transform factor
  p_package->transform_scale = scale;

  return true;
}

bool MobileSam::PromptBoxPreProcess(ParsingType package)
{
  auto p_package = std::dynamic_pointer_cast<SamPipelinePackage>(package);
  CHECK_STATE(p_package != nullptr,
              "[MobileSam Prompt PreProcess] the `package` instance \
                          is not a instance of `SamPipelinePackage`!");

  // 0. Get the decoder and encoder buffer
  auto decoder_map_blob2ptr = p_package->mask_decoder_blobs_buffer;

  auto        encoder_map_blob2ptr = p_package->image_encoder_blobs_buffer;
  const auto &encoder_output     = encoder_map_blob2ptr->GetOuterBlobBuffer(encoder_blob_names_[1]);
  void       *image_features_ptr = encoder_output.first;

  ////////////////// Transpose if decoder is rknn framework //////////////////
  if (mask_boxes_decoder_core_->GetType() == inference_core::InferCoreType::RKNN)
  {
    LOG(WARNING)
        << "[MobileSAM] Got rknn mask box decoder! Transposing Image Features to `NHWC` format!!!";
    const size_t total_image_feature_elements_num =
        IMAGE_FEATURE_HEIGHT * IMAGE_FEATURE_WIDTH * IMAGE_FEATURES_LEN;
    std::vector<float> hwc_buffer(total_image_feature_elements_num);
    rknn_nchw_2_nhwc(reinterpret_cast<float *>(encoder_output.first), hwc_buffer.data(), 1,
                     IMAGE_FEATURES_LEN, IMAGE_FEATURE_HEIGHT, IMAGE_FEATURE_WIDTH);
    memcpy(encoder_output.first, hwc_buffer.data(),
           total_image_feature_elements_num * sizeof(float));
  }
  ////////////////////////////////////////////////////////////////////////////

  // Zero-Copy Feature : let decoder use the buffer which encoder outputs
  // Encoder/Decoder with different infer_core are supported. (if the hardware support) 
  decoder_map_blob2ptr->SetBlobBuffer(
      box_dec_blob_names_[0], encoder_output.first,
      encoder_output.second); 

  // 1. Set prompt
  const auto &boxes = p_package->boxes;
  const auto &scale = p_package->transform_scale;
  float      *boxes_ptr =
      static_cast<float *>(decoder_map_blob2ptr->GetOuterBlobBuffer(box_dec_blob_names_[1]).first);
  const int64_t dynmaic_box_number = boxes.size();
  for (int i = 0; i < dynmaic_box_number; ++i)
  {
    const auto &box      = boxes[i];
    boxes_ptr[i * 4 + 0] = (box.x - box.w / 2.f) * scale;
    boxes_ptr[i * 4 + 1] = (box.y - box.h / 2.f) * scale;
    boxes_ptr[i * 4 + 2] = (box.x + box.w / 2.f) * scale;
    boxes_ptr[i * 4 + 3] = (box.y + box.h / 2.f) * scale;
  }

  // Set dynamic shape
  std::vector<int64_t> dynamic_shape{1, dynmaic_box_number, 4};
  decoder_map_blob2ptr->SetBlobShape(box_dec_blob_names_[1], dynamic_shape);

  float *mask_input =
      static_cast<float *>(decoder_map_blob2ptr->GetOuterBlobBuffer(box_dec_blob_names_[2]).first);
  memset(mask_input, 0, MASK_LOW_RES_WIDTH * MASK_LOW_RES_HEIGHT * sizeof(float));

  float *has_mask_input =
      static_cast<float *>(decoder_map_blob2ptr->GetOuterBlobBuffer(box_dec_blob_names_[3]).first);
  has_mask_input[0] = 1.f;

  // 2. Set inference buffer
  p_package->infer_buffer = decoder_map_blob2ptr;

  return true;
}

bool MobileSam::PromptPointPreProcess(ParsingType package)
{
  auto p_package = std::dynamic_pointer_cast<SamPipelinePackage>(package);
  CHECK_STATE(p_package != nullptr,
              "[MobileSam Prompt PreProcess] the `package` instance \
                          is not a instance of `SamPipelinePackage`!");

  // 0. Get the decoder and encoder buffer
  auto decoder_map_blob2ptr = p_package->mask_decoder_blobs_buffer;

  // 1. 设置image embeddings缓存指针
  auto        encoder_map_blob2ptr = p_package->image_encoder_blobs_buffer;
  const auto &encoder_output     = encoder_map_blob2ptr->GetOuterBlobBuffer(encoder_blob_names_[1]);
  void       *image_features_ptr = encoder_output.first;
  ////////////////// Transpose if decoder is rknn framework //////////////////
  if (mask_points_decoder_core_->GetType() == inference_core::InferCoreType::RKNN)
  {
    LOG(WARNING) << "[MobileSAM] Got rknn mask point decoder! Transposing Image Features to `NHWC` "
                    "format!!!";
    const size_t total_image_feature_elements_num =
        IMAGE_FEATURE_HEIGHT * IMAGE_FEATURE_WIDTH * IMAGE_FEATURES_LEN;
    std::vector<float> hwc_buffer(total_image_feature_elements_num);
    rknn_nchw_2_nhwc(reinterpret_cast<float *>(encoder_output.first), hwc_buffer.data(), 1,
                     IMAGE_FEATURES_LEN, IMAGE_FEATURE_HEIGHT, IMAGE_FEATURE_WIDTH);
    memcpy(encoder_output.first, hwc_buffer.data(),
           total_image_feature_elements_num * sizeof(float));
  }
  ////////////////////////////////////////////////////////////////////////////

  // Zero-Copy Feature : let decoder use the buffer which encoder outputs
  // Encoder/Decoder with different infer_core are supported. (if the hardware support) 
  decoder_map_blob2ptr->SetBlobBuffer(
      point_dec_blob_names_[0], encoder_output.first,
      encoder_output.second); 

  // 1. Set prompt
  const auto &points     = p_package->points;
  const auto &labels     = p_package->labels;
  const auto &scale      = p_package->transform_scale;
  float      *points_ptr = reinterpret_cast<float *>(
      decoder_map_blob2ptr->GetOuterBlobBuffer(point_dec_blob_names_[1]).first);
  float *labels_ptr = reinterpret_cast<float *>(
      decoder_map_blob2ptr->GetOuterBlobBuffer(point_dec_blob_names_[2]).first);

  const int64_t dynamic_point_number = points.size();
  for (int i = 0; i < dynamic_point_number; ++i)
  {
    const auto &point     = points[i];
    const auto &lab       = labels[i];
    points_ptr[i * 2 + 0] = static_cast<float>(point.first * scale);
    points_ptr[i * 2 + 1] = static_cast<float>(point.second * scale);
    labels_ptr[i]         = static_cast<float>(lab);
  }

  // Set dynamic shape
  std::vector<int64_t> coords_dynamic_shape{1, dynamic_point_number, 2};
  decoder_map_blob2ptr->SetBlobShape(point_dec_blob_names_[1], coords_dynamic_shape);
  std::vector<int64_t> labels_dynamic_shape{1, dynamic_point_number};
  decoder_map_blob2ptr->SetBlobShape(point_dec_blob_names_[2], labels_dynamic_shape);

  float *mask_input = static_cast<float *>(
      decoder_map_blob2ptr->GetOuterBlobBuffer(point_dec_blob_names_[3]).first);
  memset(mask_input, 0, MASK_LOW_RES_HEIGHT * MASK_LOW_RES_WIDTH * sizeof(float));

  float *has_mask_input = static_cast<float *>(
      decoder_map_blob2ptr->GetOuterBlobBuffer(point_dec_blob_names_[4]).first);
  has_mask_input[0] = 1.f;

  // 2. Set inference buffer
  p_package->infer_buffer = decoder_map_blob2ptr;

  return true;
}

bool MobileSam::MaskPostProcess(ParsingType package)
{
  auto p_package = std::dynamic_pointer_cast<SamPipelinePackage>(package);
  CHECK_STATE(p_package != nullptr,
              "[MobileSam Mask PostProcess] the `package` instance \
                          is not a instance of `SamPipelinePackage`!");

  auto decoder_map_blob2ptr = p_package->mask_decoder_blobs_buffer;

  // 1. Get the output masks buffer
  void *decoder_output_masks_ptr =
      decoder_map_blob2ptr->GetOuterBlobBuffer(MASK_OUT_BLOB_NAME).first;
  cv::Mat masks_output(MASK_LOW_RES_HEIGHT, MASK_LOW_RES_WIDTH, CV_32FC1, decoder_output_masks_ptr);

  // 2. resize to 1024,1024
  cv::resize(masks_output, masks_output, {IMAGE_INPUT_WIDTH, IMAGE_INPUT_HEIGHT});

  // 3. crop valid block
  const auto &input_image_info = p_package->input_image_data->GetImageDataInfo();
  const auto &scale            = p_package->transform_scale;
  masks_output                 = masks_output(cv::Range(0, input_image_info.image_height * scale),
                                              cv::Range(0, input_image_info.image_width * scale));

  // 4. resize to original size
  cv::resize(masks_output, masks_output,
             {input_image_info.image_width, input_image_info.image_height});

  // 5. convert to binary mask
  cv::threshold(masks_output, masks_output, 0, 1, cv::THRESH_BINARY);

  // 6. convert to CV_8U
  masks_output = masks_output * 255;
  masks_output.convertTo(masks_output, CV_8U);

  p_package->mask = masks_output;

  return true;
}

std::shared_ptr<BaseSamModel> CreateMobileSamModel(
    std::shared_ptr<inference_core::BaseInferCore>      image_encoder_core,
    std::shared_ptr<inference_core::BaseInferCore>      mask_points_decoder_core,
    std::shared_ptr<inference_core::BaseInferCore>      mask_boxes_decoder_core,
    std::shared_ptr<detection_2d::IDetectionPreProcess> image_preprocess_block,
    const std::vector<std::string>                     &encoder_blob_names,
    const std::vector<std::string>                     &box_dec_blob_names,
    const std::vector<std::string>                     &point_dec_blob_names)
{
  return std::make_shared<MobileSam>(image_encoder_core, mask_points_decoder_core,
                                     mask_boxes_decoder_core, image_preprocess_block,
                                     encoder_blob_names, box_dec_blob_names, point_dec_blob_names);
}


} // namespace sam