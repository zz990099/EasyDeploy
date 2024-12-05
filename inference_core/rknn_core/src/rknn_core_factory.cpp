/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-12-02 19:43:23
 * @LastEditTime: 2024-12-02 19:50:55
 * @FilePath: /easy_deploy/inference_core/rknn_core/src/rknn_core_factory.cpp
 */
#include "rknn_core/rknn_core.h"

namespace inference_core {

struct RknnInferCoreParams {
  std::string                                          model_path;
  std::unordered_map<std::string, RknnInputTensorType> map_blob_type;
  int                                                  mem_buf_size;
  int                                                  parallel_ctx_num;
};

class RknnInferCoreFactory : public BaseInferCoreFactory {
public:
  RknnInferCoreFactory(const RknnInferCoreParams &params) : params_(params)
  {}

  std::shared_ptr<BaseInferCore> Create() override
  {
    return CreateRknnInferCore(params_.model_path, params_.map_blob_type, params_.mem_buf_size,
                               params_.parallel_ctx_num);
  }

private:
  const RknnInferCoreParams params_;
};

std::shared_ptr<BaseInferCoreFactory> CreateRknnInferCoreFactory(
    std::string                                                 model_path,
    const std::unordered_map<std::string, RknnInputTensorType> &map_blob_type,
    const int                                                   mem_buf_size,
    const int                                                   parallel_ctx_num)
{
  RknnInferCoreParams params;
  params.model_path       = model_path;
  params.map_blob_type    = map_blob_type;
  params.mem_buf_size     = mem_buf_size;
  params.parallel_ctx_num = parallel_ctx_num;

  return std::make_shared<RknnInferCoreFactory>(params);
}

} // namespace inference_core