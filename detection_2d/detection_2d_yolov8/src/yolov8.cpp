/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-26 19:41:34
 * @LastEditTime: 2024-12-03 15:57:46
 * @FilePath: /EasyDeploy/detection_2d/detection_2d_yolov8/src/yolov8.cpp
 */
#include "detection_2d_yolov8/yolov8.h"

namespace detection_2d {

class Yolov8Detection : public BaseDetectionModel {
public:
  Yolov8Detection(const std::shared_ptr<inference_core::BaseInferCore> &infer_core,
                  const std::shared_ptr<IDetectionPreProcess>          &preprocess_block,
                  const std::shared_ptr<IDetectionPostProcess>         &postprocess_block,
                  const int                                             input_height,
                  const int                                             input_width,
                  const int                                             input_channel,
                  const int                                             cls_number,
                  const std::vector<std::string>                       &input_blobs_name,
                  const std::vector<std::string>                       &output_blobs_name,
                  const std::vector<int> downsample_scales = {8, 16, 32});

  ~Yolov8Detection() = default;

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

  const std::vector<int> downsample_scales_;

  const std::shared_ptr<inference_core::BaseInferCore> infer_core_;
  std::shared_ptr<IDetectionPreProcess>                preprocess_block_;
  std::shared_ptr<IDetectionPostProcess>               postprocess_block_;
};

Yolov8Detection::Yolov8Detection(const std::shared_ptr<inference_core::BaseInferCore> &infer_core,
                                 const std::shared_ptr<IDetectionPreProcess>  &preprocess_block,
                                 const std::shared_ptr<IDetectionPostProcess> &postprocess_block,
                                 const int                                     input_height,
                                 const int                                     input_width,
                                 const int                                     input_channel,
                                 const int                                     cls_number,
                                 const std::vector<std::string>               &input_blobs_name,
                                 const std::vector<std::string>               &output_blobs_name,
                                 const std::vector<int>                        downsample_scales)
    : BaseDetectionModel(infer_core),
      infer_core_(infer_core),
      preprocess_block_(preprocess_block),
      postprocess_block_(postprocess_block),
      input_height_(input_height),
      input_width_(input_width),
      input_channel_(input_channel),
      cls_number_(cls_number),
      input_blobs_name_(input_blobs_name),
      output_blobs_name_(output_blobs_name),
      downsample_scales_(downsample_scales)
{
  // Check if the input arguments and inference_core matches
  auto blobs_tensor = infer_core_->AllocBlobsBuffer();
  if (blobs_tensor->Size() != input_blobs_name_.size() + output_blobs_name_.size())
  {
    LOG(ERROR) << "[Yolov8Detection] Infer core should has {"
               << input_blobs_name_.size() + output_blobs_name_.size() << "} blobs !"
               << " but got " << blobs_tensor->Size() << " blobs";
    throw std::runtime_error("[Yolov8Detection] Got invalid input arguments!!");
  }

  for (const std::string &input_blob_name : input_blobs_name)
  {
    blobs_tensor->GetTensor(input_blob_name);
  }

  for (const std::string &output_blob_name : output_blobs_name)
  {
    blobs_tensor->GetTensor(output_blob_name);
  }

  for (const int s : downsample_scales_)
  {
    if (input_height_ % s != 0)
    {
      LOG(ERROR) << "[Yolov8Detection] `input_height_` should be an integer multiple of `s`";
      throw std::runtime_error("[Yolov8Detection] Got invalid input arguments!!");
    }
    if (input_width_ % s != 0)
    {
      LOG(ERROR) << "[Yolov8Detection] `input_width_` should be an integer multiple of `s`";
      throw std::runtime_error("[Yolov8Detection] Got invalid input arguments!!");
    }
  }
}

bool Yolov8Detection::PreProcess(std::shared_ptr<async_pipeline::IPipelinePackage> _package)
{
  auto package = std::dynamic_pointer_cast<DetectionPipelinePackage>(_package);
  CHECK_STATE(package != nullptr,
              "[Yolov8Detection] PreProcess the `_package` instance does not belong to "
              "`DetectionPipelinePackage`");

  const auto &blobs_tensor = package->GetInferBuffer();

  float scale = preprocess_block_->Preprocess(package->input_image_data,
                                              blobs_tensor->GetTensor(input_blobs_name_[0]),
                                              input_height_, input_width_);

  package->transform_scale = scale;
  return true;
}

bool Yolov8Detection::PostProcess(std::shared_ptr<async_pipeline::IPipelinePackage> _package)
{
  auto package = std::dynamic_pointer_cast<DetectionPipelinePackage>(_package);
  CHECK_STATE(package != nullptr,
              "[Yolov8Detection] PostProcess the `_package` instance does not belong to "
              "`DetectionPipelinePackage`");

  auto                p_blob_buffers = package->GetInferBuffer();
  std::vector<void *> output_blobs_ptr;
  for (const std::string &output_blob_name : output_blobs_name_)
  {
    void *blob_ptr = p_blob_buffers->GetTensor(output_blob_name)->RawPtr();
    output_blobs_ptr.push_back(blob_ptr);
  }

  postprocess_block_->Postprocess(output_blobs_ptr,
                                  package->results, // mutable
                                  package->conf_thresh, package->transform_scale);
  return true;
}

std::shared_ptr<BaseDetectionModel> CreateYolov8DetectionModel(
    const std::shared_ptr<inference_core::BaseInferCore> &infer_core,
    const std::shared_ptr<IDetectionPreProcess>          &preprocess_block,
    const std::shared_ptr<IDetectionPostProcess>         &postprocess_block,
    const int                                             input_height,
    const int                                             input_width,
    const int                                             input_channel,
    const int                                             cls_number,
    const std::vector<std::string>                       &input_blobs_name,
    const std::vector<std::string>                       &output_blobs_name,
    const std::vector<int>                                downsample_scales)
{
  return std::make_shared<Yolov8Detection>(infer_core, preprocess_block, postprocess_block,
                                           input_height, input_width, input_channel, cls_number,
                                           input_blobs_name, output_blobs_name, downsample_scales);
}

} // namespace detection_2d
