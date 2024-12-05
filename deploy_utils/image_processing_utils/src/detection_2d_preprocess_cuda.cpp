/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-19 18:33:00
 * @LastEditTime: 2024-12-02 20:14:47
 * @FilePath: /easy_deploy/deploy_utils/image_processing_utils/src/detection_2d_preprocess_cuda.cpp
 */
#include "detection_2d_util/detection_2d_util.h"

#include <cuda_runtime.h>

extern "C" float CallCudaPreprocess(const uint8_t *src,
                                    int            src_width,
                                    int            src_height,
                                    float         *dst,
                                    int            dst_width,
                                    int            dst_height,
                                    void          *unified_mem_buffer);

namespace detection_2d {

class DetPreProcessCUDA : public IDetectionPreProcess {
public:
  DetPreProcessCUDA(const int max_src_height   = 1920,
                    const int max_src_width    = 1920,
                    const int max_src_channels = 3);

  float Preprocess(std::shared_ptr<async_pipeline::IPipelineImageData> input_image_data,
                   std::shared_ptr<inference_core::IBlobsBuffer>       blob_buffer,
                   const std::string                                  &blob_name,
                   int                                                 dst_height,
                   int                                                 dst_width) override;

  ~DetPreProcessCUDA();

private:
  float CudaPreprocess(
      const uint8_t *src, int src_width, int src_height, float *dst, int dst_width, int dst_height);

private:
  const int max_src_width_;
  const int max_src_height_;
  void     *device_mem_buffer_ = nullptr;
};

DetPreProcessCUDA::DetPreProcessCUDA(const int max_src_height,
                                     const int max_src_width,
                                     const int src_channels)
    : max_src_height_(max_src_height), max_src_width_(max_src_width)
{
  const int max_input_byte_size = max_src_height_ * max_src_width_ * src_channels;

  auto malloc_ret = cudaMalloc(&device_mem_buffer_, max_input_byte_size);
  if (malloc_ret != cudaSuccess)
  {
    throw std::runtime_error("[DetPreProcessCUDA] CudaMalloc failed to alloc memory on cuda!!!");
  }
}

float DetPreProcessCUDA::Preprocess(
    std::shared_ptr<async_pipeline::IPipelineImageData> input_image_data,
    std::shared_ptr<inference_core::IBlobsBuffer>       blob_buffer,
    const std::string                                  &blob_name,
    int                                                 dst_height,
    int                                                 dst_width)
{
  // 1. Make sure the buffer ptr is on device side
  blob_buffer->SetBlobBuffer(blob_name, DataLocation::DEVICE);
  auto _dst_ptr = blob_buffer->GetOuterBlobBuffer(blob_name);

  float      *dst_ptr         = static_cast<float *>(_dst_ptr.first);
  const auto &image_data_info = input_image_data->GetImageDataInfo();

  // 2. Call cuda kernel function
  return CudaPreprocess(image_data_info.data_pointer, image_data_info.image_width,
                        image_data_info.image_height, dst_ptr, dst_width, dst_height);
}

float DetPreProcessCUDA::CudaPreprocess(
    const uint8_t *src, int src_width, int src_height, float *dst, int dst_width, int dst_height)
{
  return CallCudaPreprocess(src, src_width, src_height, dst, dst_width, dst_height,
                            device_mem_buffer_);
}

DetPreProcessCUDA::~DetPreProcessCUDA()
{
  if (device_mem_buffer_ != nullptr)
  {
    cudaFree(device_mem_buffer_);
    device_mem_buffer_ = nullptr;
  }
}

std::shared_ptr<IDetectionPreProcess> CreateCudaDetPreProcess(const int max_src_height,
                                                              const int max_src_width,
                                                              const int max_src_channels)
{
  return std::make_shared<DetPreProcessCUDA>(max_src_height, max_src_width, max_src_channels);
}

struct Detection2DPreprocessCudaParams {
  int max_src_height;
  int max_src_width;
  int max_src_channels;
};

class Detection2DPreprocessCudaFactory : public BaseDetectionPreprocessFactory {
public:
  Detection2DPreprocessCudaFactory(const Detection2DPreprocessCudaParams &params) : params_(params)
  {}

  std::shared_ptr<IDetectionPreProcess> Create() override
  {
    return CreateCudaDetPreProcess(params_.max_src_height, params_.max_src_width,
                                   params_.max_src_channels);
  }

private:
  const Detection2DPreprocessCudaParams params_;
};

std::shared_ptr<BaseDetectionPreprocessFactory> CreateCudaDetPreProcessFactory(
    const int max_src_height, const int max_src_width, const int max_src_channels)
{
  Detection2DPreprocessCudaParams params;
  params.max_src_height   = max_src_height;
  params.max_src_width    = max_src_width;
  params.max_src_channels = max_src_channels;

  return std::make_shared<Detection2DPreprocessCudaFactory>(params);
}

} // namespace detection_2d