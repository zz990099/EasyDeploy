/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-19 18:33:00
 * @LastEditTime: 2024-12-02 20:14:25
 * @FilePath: /easy_deploy/deploy_utils/image_processing_utils/src/detection_2d_preprocess_cpu.cpp
 */
#include "detection_2d_util/detection_2d_util.h"

namespace detection_2d {

class DetPreProcessCPU : public IDetectionPreProcess {
public:
  DetPreProcessCPU(const std::vector<float> &mean,
                   const std::vector<float> &val,
                   bool                      do_transpose = true,
                   bool                      do_norm      = true);

  float Preprocess(std::shared_ptr<async_pipeline::IPipelineImageData> input_image_data,
                   std::shared_ptr<inference_core::IBlobsBuffer>       blob_buffer,
                   const std::string                                  &blob_name,
                   int                                                 dst_height,
                   int                                                 dst_width) override;

private:
  void FlipChannelsWithNorm(const cv::Mat &image, float *dst_ptr, bool flip);
  void FlipChannelsWithoutNorm(const cv::Mat &image, u_char *dst_ptr, bool flip);
  void TransposeAndFilpWithNorm(const cv::Mat &image, float *dst_ptr, bool flip);
  void TransposeAndFilpWithoutNorm(const cv::Mat &image, u_char *dst_ptr, bool flip);

private:
  const std::vector<float> mean_, val_;
  const bool               do_transpose_, do_norm_;
};

DetPreProcessCPU::DetPreProcessCPU(const std::vector<float> &mean,
                                   const std::vector<float> &val,
                                   bool                      do_transpose,
                                   bool                      do_norm)
    : mean_(mean), val_(val), do_transpose_(do_transpose), do_norm_(do_norm)
{}

float DetPreProcessCPU::Preprocess(
    std::shared_ptr<async_pipeline::IPipelineImageData> input_image_data,
    std::shared_ptr<inference_core::IBlobsBuffer>       blob_buffer,
    const std::string                                  &blob_name,
    int                                                 dst_height,
    int                                                 dst_width)
{
  // 0. Make sure read/write on the host-side memory buffer
  blob_buffer->SetBlobBuffer(blob_name, DataLocation::HOST);
  auto _dst_ptr = blob_buffer->GetOuterBlobBuffer(blob_name);

  // 1. Caculate transform factor
  float      *dst_ptr         = static_cast<float *>(_dst_ptr.first);
  const auto &image_data_info = input_image_data->GetImageDataInfo();
  const int   image_height    = image_data_info.image_height;
  const int   image_width     = image_data_info.image_width;
  int         fix_height, fix_width;
  float       scale;

  const float s_w = static_cast<float>(dst_width) / image_width;
  const float s_h = static_cast<float>(dst_height) / image_height;

  if (s_h < s_w)
  {
    fix_height = dst_height;
    scale      = s_h;
    fix_width  = static_cast<int>(image_width * scale);
  } else
  {
    fix_width  = dst_width;
    scale      = s_w;
    fix_height = static_cast<int>(image_height * scale);
  }

  // 2. rebuild the cv::Mat format image
  cv::Mat input_image(image_height, image_width, CV_8UC3, image_data_info.data_pointer);

  // 3. resize and padding to the left-top
  cv::Mat resized_image;
  cv::resize(input_image, resized_image, {fix_width, fix_height});
  cv::Mat dst_image = cv::Mat(dst_height, dst_width, CV_8UC3, cv::Scalar{0, 0, 0});
  resized_image.copyTo(dst_image(cv::Rect(0, 0, fix_width, fix_height)));

  if (!do_transpose_)
  {
    // 4. flip and norm
    if (do_norm_)
    {
      FlipChannelsWithNorm(dst_image, dst_ptr, image_data_info.format == ImageDataFormat::BGR);
    } else
    {
      FlipChannelsWithoutNorm(dst_image, reinterpret_cast<u_char *>(dst_ptr),
                              image_data_info.format == ImageDataFormat::BGR);
    }
  } else
  {
    // 5. transpose flip and norm
    if (do_norm_)
    {
      TransposeAndFilpWithNorm(dst_image, dst_ptr, image_data_info.format == ImageDataFormat::BGR);
    } else
    {
      TransposeAndFilpWithoutNorm(dst_image, reinterpret_cast<u_char *>(dst_ptr),
                                  image_data_info.format == ImageDataFormat::BGR);
    }
  }

  return scale;
}

void DetPreProcessCPU::FlipChannelsWithNorm(const cv::Mat &image, float *dst_ptr, bool flip)
{
  const int rows = image.rows, cols = image.cols;

  const int r_idx = flip ? 2 : 0;
  const int b_idx = flip ? 0 : 2;
  const int g_idx = 1;

  for (int r = 0; r < rows; ++r)
  {
    size_t offset = (r * cols) * 3;
    for (int c = 0; c < cols; ++c)
    {
      size_t idx       = offset + c * 3;
      dst_ptr[idx + 0] = (image.data[idx + r_idx] - mean_[r_idx]) / val_[r_idx];
      dst_ptr[idx + 1] = (image.data[idx + g_idx] - mean_[g_idx]) / val_[g_idx];
      dst_ptr[idx + 2] = (image.data[idx + b_idx] - mean_[b_idx]) / val_[b_idx];
    }
  }
}

void DetPreProcessCPU::FlipChannelsWithoutNorm(const cv::Mat &image, u_char *dst_ptr, bool flip)
{
  const int rows = image.rows, cols = image.cols;

  const int r_idx = flip ? 2 : 0;
  const int b_idx = flip ? 0 : 2;
  const int g_idx = 1;

  for (int r = 0; r < rows; ++r)
  {
    size_t offset = r * cols * 3;
    for (int c = 0; c < cols; ++c)
    {
      size_t idx       = offset + c * 3;
      dst_ptr[idx + 0] = image.data[idx + r_idx];
      dst_ptr[idx + 1] = image.data[idx + g_idx];
      dst_ptr[idx + 2] = image.data[idx + b_idx];
    }
  }
}

void DetPreProcessCPU::TransposeAndFilpWithNorm(const cv::Mat &image, float *dst_ptr, bool flip)
{
  const int rows = image.rows, cols = image.cols;

  const int r_idx = flip ? 2 : 0;
  const int b_idx = flip ? 0 : 2;
  const int g_idx = 1;

  const int single_channel_pixel_size = rows * cols;
  for (int r = 0; r < rows; ++r)
  {
    uchar *pixel_ptr    = image.data + r * image.step;
    int    offset_start = cols * r;
    for (int c = 0; c < cols; ++c)
    {
      int offset_idx = offset_start + c;

      dst_ptr[offset_idx + 0 * single_channel_pixel_size] =
          (pixel_ptr[r_idx] - mean_[r_idx]) / val_[r_idx];

      dst_ptr[offset_idx + 1 * single_channel_pixel_size] =
          (pixel_ptr[g_idx] - mean_[g_idx]) / val_[g_idx];

      dst_ptr[offset_idx + 2 * single_channel_pixel_size] =
          (pixel_ptr[b_idx] - mean_[b_idx]) / val_[b_idx];

      pixel_ptr += 3;
    }
  }
}

void DetPreProcessCPU::TransposeAndFilpWithoutNorm(const cv::Mat &image, u_char *dst_ptr, bool flip)
{
  const int rows = image.rows, cols = image.cols;

  const int r_idx = flip ? 2 : 0;
  const int b_idx = flip ? 0 : 2;
  const int g_idx = 1;

  const int single_channel_pixel_size = rows * cols;
  for (int r = 0; r < rows; ++r)
  {
    uchar *pixel_ptr    = image.data + r * image.step;
    int    offset_start = cols * r;
    for (int c = 0; c < cols; ++c)
    {
      int offset_idx = offset_start + c;

      dst_ptr[offset_idx + 0 * single_channel_pixel_size] = pixel_ptr[r_idx];
      dst_ptr[offset_idx + 1 * single_channel_pixel_size] = pixel_ptr[g_idx];
      dst_ptr[offset_idx + 2 * single_channel_pixel_size] = pixel_ptr[b_idx];

      pixel_ptr += 3;
    }
  }
}

std::shared_ptr<IDetectionPreProcess> CreateCpuDetPreProcess(const std::vector<float> &mean,
                                                             const std::vector<float> &val,
                                                             bool                      do_transpose,
                                                             bool                      do_norm)
{
  return std::make_shared<DetPreProcessCPU>(mean, val, do_transpose, do_norm);
}

struct Detection2DPreprocessCpuParams {
  std::vector<float> mean;
  std::vector<float> val;
  bool               do_transpose;
  bool               do_norm;
};

class Detection2DPreprocessCpuFactory : public BaseDetectionPreprocessFactory {
public:
  Detection2DPreprocessCpuFactory(const Detection2DPreprocessCpuParams &params) : params_(params)
  {}

  std::shared_ptr<IDetectionPreProcess> Create() override
  {
    return CreateCpuDetPreProcess(params_.mean, params_.val, params_.do_transpose, params_.do_norm);
  }

private:
  const Detection2DPreprocessCpuParams params_;
};

std::shared_ptr<BaseDetectionPreprocessFactory> CreateCpuDetPreProcessFactory(
    const std::vector<float> &mean, const std::vector<float> &val, bool do_transpose, bool do_norm)
{
  Detection2DPreprocessCpuParams params;
  params.mean         = mean;
  params.val          = val;
  params.do_transpose = do_transpose;
  params.do_norm      = do_norm;

  return std::make_shared<Detection2DPreprocessCpuFactory>(params);
}

} // namespace detection_2d
