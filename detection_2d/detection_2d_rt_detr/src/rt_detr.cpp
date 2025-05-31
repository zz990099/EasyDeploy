#include "detection_2d_rt_detr/rt_detr.h"

namespace detection_2d {

class RTDetrDetection : public BaseDetectionModel {
public:
  RTDetrDetection(const std::shared_ptr<inference_core::BaseInferCore> &infer_core,
                           const std::shared_ptr<IDetectionPreProcess>          &preprocess_block,
                           const int                                             input_height,
                           const int                                             input_width,
                           const int                                             input_channel,
                           const int                                             cls_number,
                           const std::vector<std::string>                       &input_blobs_name,
                           const std::vector<std::string>                       &output_blobs_name);

  ~RTDetrDetection() = default;

private:
  bool PreProcess(std::shared_ptr<async_pipeline::IPipelinePackage> pipeline_unit) override;

  bool PostProcess(std::shared_ptr<async_pipeline::IPipelinePackage> pipeline_unit) override;

private:
  const std::vector<std::string> input_blobs_name_;
  const std::vector<std::string> output_blobs_name_;
  const int                      input_height_;
  const int                      input_width_;
  const int                      input_channel_;
  const int                      cls_number_;

  const std::shared_ptr<inference_core::BaseInferCore> infer_core_;
  std::shared_ptr<IDetectionPreProcess>                preprocess_block_;
};

RTDetrDetection::RTDetrDetection(
    const std::shared_ptr<inference_core::BaseInferCore> &infer_core,
    const std::shared_ptr<IDetectionPreProcess>          &preprocess_block,
    const int                                             input_height,
    const int                                             input_width,
    const int                                             input_channel,
    const int                                             cls_number,
    const std::vector<std::string>                       &input_blobs_name,
    const std::vector<std::string>                       &output_blobs_name)
    : BaseDetectionModel(infer_core),
      input_blobs_name_(input_blobs_name),
      output_blobs_name_(output_blobs_name),
      input_height_(input_height),
      input_width_(input_width),
      input_channel_(input_channel),
      cls_number_(cls_number),
      infer_core_(infer_core),
      preprocess_block_(preprocess_block)
{
  // 创建并获取一个缓存句柄，用于校验模型与算法的一致性
  auto p_map_buffer2ptr = infer_core_->AllocBlobsBuffer();
  if (p_map_buffer2ptr->Size() != input_blobs_name_.size() + output_blobs_name_.size())
  {
    LOG(ERROR) << "[RTDetrDetection] Infer core should has {"
               << input_blobs_name_.size() + output_blobs_name_.size() << "} blobs !"
               << " but got " << p_map_buffer2ptr->Size() << " blobs";
    throw std::runtime_error(
        "[RTDetrDetection] Construction Failed!!! Got invalid blobs_num size!!!");
  }

  for (const std::string &input_blob_name : input_blobs_name)
  {
    if (p_map_buffer2ptr->GetOuterBlobBuffer(input_blob_name).first == nullptr)
    {
      LOG(ERROR) << "[RTDetrDetection] Input_blob_name_ {" << input_blob_name
                 << "input blob name does not match `infer_core_` !";
      throw std::runtime_error(
          "[RTDetrDetection] Construction Failed!!! Got invalid input_blob_name!!!");
    }
  }

  for (const std::string &output_blob_name : output_blobs_name)
  {
    if (p_map_buffer2ptr->GetOuterBlobBuffer(output_blob_name).first == nullptr)
    {
      LOG(ERROR) << "[RTDetrDetection] Output_blob_name_ {" << output_blob_name
                 << "output blob name does not match `infer_core_` !";
      throw std::runtime_error(
          "[RTDetrDetection] Construction Failed!!! Got invalid output_blob_name!!!");
    }
  }
}

bool RTDetrDetection::PreProcess(
    std::shared_ptr<async_pipeline::IPipelinePackage> _package)
{
  auto package = std::dynamic_pointer_cast<DetectionPipelinePackage>(_package);
  CHECK_STATE(package != nullptr,
              "[RTDetrDetection] PreProcess the `_package` instance does not belong to "
              "`DetectionPipelinePackage`");

  const auto &blobs_buffer = package->GetInferBuffer();
  float       scale        = preprocess_block_->Preprocess(package->input_image_data, blobs_buffer,
                                                           input_blobs_name_[0], input_height_, input_width_);
  package->transform_scale = scale;
  return true;
}

bool RTDetrDetection::PostProcess(
    std::shared_ptr<async_pipeline::IPipelinePackage> _package)
{
  auto package = std::dynamic_pointer_cast<DetectionPipelinePackage>(_package);
  CHECK_STATE(package != nullptr,
              "[RTDetrDetection] PostProcess the `_package` instance does not belong to "
              "`DetectionPipelinePackage`");

  const auto &blobs_buffer = package->GetInferBuffer();

  // RTDetrDetection outputs: labels <int32> (1, 300); boxes (1, 300, 4); scores (1, 300)

  float *labels_ptr = reinterpret_cast<float *>(blobs_buffer->GetOuterBlobBuffer("labels").first);
  float *boxes_ptr  = reinterpret_cast<float *>(blobs_buffer->GetOuterBlobBuffer("boxes").first);
  float *scores_ptr = reinterpret_cast<float *>(blobs_buffer->GetOuterBlobBuffer("scores").first);

  const int   CANDIDATES_NUM = 300;
  const float conf_thresh    = package->conf_thresh;
  const float transf_scale   = package->transform_scale;

  std::vector<BBox2D> valid_boxes;
  for (int i = 0; i < CANDIDATES_NUM; ++i)
  {
    if (scores_ptr[i] >= conf_thresh)
    {
      float x0 = boxes_ptr[i * 4 + 0];
      float y0 = boxes_ptr[i * 4 + 1];
      float x1 = boxes_ptr[i * 4 + 2];
      float y1 = boxes_ptr[i * 4 + 3];

      BBox2D box;
      box.x    = (x0 + x1) / 2 / transf_scale;
      box.y    = (y0 + y1) / 2 / transf_scale;
      box.w    = (x1 - x0) / transf_scale;
      box.h    = (y1 - y0) / transf_scale;
      box.cls  = static_cast<float>(labels_ptr[i]);
      box.conf = scores_ptr[i];
      valid_boxes.push_back(box);
    }
  }

  package->results = std::move(valid_boxes);

  return true;
}

std::shared_ptr<BaseDetectionModel> CreateRTDetrDetectionModel(
    const std::shared_ptr<inference_core::BaseInferCore> &infer_core,
    const std::shared_ptr<IDetectionPreProcess>          &preprocess_block,
    const int                                             input_height,
    const int                                             input_width,
    const int                                             input_channel,
    const int                                             cls_number,
    const std::vector<std::string>                       &input_blobs_name,
    const std::vector<std::string>                       &output_blobs_name)
{
  return std::make_shared<RTDetrDetection>(infer_core, preprocess_block, input_height,
                                                    input_width, input_channel, cls_number,
                                                    input_blobs_name, output_blobs_name);
}

} // namespace detection_2d
