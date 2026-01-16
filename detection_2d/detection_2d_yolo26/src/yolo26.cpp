#include "detection_2d_yolo26/yolo26.hpp"

namespace easy_deploy {

class Yolo26Detection : public BaseDetectionModel {
public:
  Yolo26Detection(const std::shared_ptr<BaseInferCore>        &infer_core,
                  const std::shared_ptr<IDetectionPreProcess> &preprocess_block,
                  const int                                    input_height,
                  const int                                    input_width,
                  const int                                    input_channel,
                  const int                                    cls_number,
                  const std::vector<std::string>              &input_blobs_name,
                  const std::vector<std::string>              &output_blobs_name);

  ~Yolo26Detection() = default;

private:
  bool PreProcess(std::shared_ptr<IPipelinePackage> pipeline_unit) override;

  bool PostProcess(std::shared_ptr<IPipelinePackage> pipeline_unit) override;

private:
  const std::vector<std::string> input_blobs_name_;
  const std::vector<std::string> output_blobs_name_;
  const int                      input_height_;
  const int                      input_width_;
  const int                      input_channel_;
  const int                      cls_number_;

  const std::shared_ptr<BaseInferCore>  infer_core_;
  std::shared_ptr<IDetectionPreProcess> preprocess_block_;
};

Yolo26Detection::Yolo26Detection(const std::shared_ptr<BaseInferCore>        &infer_core,
                                 const std::shared_ptr<IDetectionPreProcess> &preprocess_block,
                                 const int                                    input_height,
                                 const int                                    input_width,
                                 const int                                    input_channel,
                                 const int                                    cls_number,
                                 const std::vector<std::string>              &input_blobs_name,
                                 const std::vector<std::string>              &output_blobs_name)
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
  auto blobs_tensor = infer_core_->AllocBlobsBuffer();
  if (blobs_tensor->Size() != input_blobs_name_.size() + output_blobs_name_.size())
  {
    LOG_ERROR("[Yolo26Detection] Infer core should has {%ld} blobs, but got {%ld} blobs",
              input_blobs_name_.size() + output_blobs_name_.size(), blobs_tensor->Size());
    throw std::runtime_error(
        "[Yolo26Detection] Construction Failed!!! Got invalid blobs_num size!!!");
  }

  for (const std::string &input_blob_name : input_blobs_name)
  {
    blobs_tensor->GetTensor(input_blob_name);
  }

  for (const std::string &output_blob_name : output_blobs_name)
  {
    blobs_tensor->GetTensor(output_blob_name);
  }
}

bool Yolo26Detection::PreProcess(std::shared_ptr<IPipelinePackage> _package)
{
  auto package = std::dynamic_pointer_cast<DetectionPipelinePackage>(_package);
  CHECK_STATE(package != nullptr,
              "[Yolo26Detection] PreProcess the `_package` instance does not belong to "
              "`DetectionPipelinePackage`");

  const auto &blobs_tensor = package->GetInferBuffer();

  float scale = preprocess_block_->Preprocess(package->input_image_data,
                                              blobs_tensor->GetTensor(input_blobs_name_[0]),
                                              input_height_, input_width_);

  package->transform_scale = scale;
  return true;
}

bool Yolo26Detection::PostProcess(std::shared_ptr<IPipelinePackage> _package)
{
  auto package = std::dynamic_pointer_cast<DetectionPipelinePackage>(_package);
  CHECK_STATE(package != nullptr,
              "[Yolo26Detection] PostProcess the `_package` instance does not belong to "
              "`DetectionPipelinePackage`");

  const auto &blobs_tensor = package->GetInferBuffer();

  // Yolo26Detection outputs: <float32> (1, 300, 6): conf(1), label(1), box(4)

  float *output_ptr = blobs_tensor->GetTensor(output_blobs_name_[0])->Cast<float>();

  const int   CANDIDATES_NUM = 300;
  const float conf_thresh    = package->conf_thresh;
  const float transf_scale   = package->transform_scale;

  std::vector<BBox2D> valid_boxes;
  for (int i = 0; i < CANDIDATES_NUM; ++i)
  {
    const int offset = 6 * i;
    if (output_ptr[offset + 4] >= conf_thresh)
    {
      float x0 = output_ptr[offset + 0];
      float y0 = output_ptr[offset + 1];
      float x1 = output_ptr[offset + 2];
      float y1 = output_ptr[offset + 3];

      BBox2D box;
      box.x    = (x0 + x1) / 2 / transf_scale;
      box.y    = (y0 + y1) / 2 / transf_scale;
      box.w    = (x1 - x0) / transf_scale;
      box.h    = (y1 - y0) / transf_scale;
      box.cls  = static_cast<float>(output_ptr[offset + 5]);
      box.conf = output_ptr[offset + 4];
      valid_boxes.push_back(box);
    }
  }

  package->results = std::move(valid_boxes);

  return true;
}

std::shared_ptr<BaseDetectionModel> CreateYolo26DetectionModel(
    const std::shared_ptr<BaseInferCore>        &infer_core,
    const std::shared_ptr<IDetectionPreProcess> &preprocess_block,
    const int                                    input_height,
    const int                                    input_width,
    const int                                    input_channel,
    const int                                    cls_number,
    const std::vector<std::string>              &input_blobs_name,
    const std::vector<std::string>              &output_blobs_name)
{
  return std::make_shared<Yolo26Detection>(infer_core, preprocess_block, input_height, input_width,
                                           input_channel, cls_number, input_blobs_name,
                                           output_blobs_name);
}

} // namespace easy_deploy
