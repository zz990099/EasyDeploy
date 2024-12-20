/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-19 18:33:05
 * @LastEditTime: 2024-12-02 19:43:10
 * @FilePath: /easy_deploy/inference_core/rknn_core/include/rknn_core/rknn_core.h
 */
#ifndef __EASY_DEPLOY_INFERENCE_CORE_RKNN_CORE_H
#define __EASY_DEPLOY_INFERENCE_CORE_RKNN_CORE_H

#include "deploy_core/base_infer_core.h"

namespace inference_core {

enum RknnInputTensorType {
  RK_UINT8,
  RK_INT8,
  RK_FLOAT16,
  RK_FLOAT32,
  RK_UINT32,
  RK_INT32,
  RK_INT64
};

std::shared_ptr<BaseInferCore> CreateRknnInferCore(
    std::string                                                 model_path,
    const std::unordered_map<std::string, RknnInputTensorType> &map_blob_type    = {},
    const int                                                   mem_buf_size     = 5,
    const int                                                   parallel_ctx_num = 1);

std::shared_ptr<BaseInferCoreFactory> CreateRknnInferCoreFactory(
    std::string                                                 model_path,
    const std::unordered_map<std::string, RknnInputTensorType> &map_blob_type    = {},
    const int                                                   mem_buf_size     = 5,
    const int                                                   parallel_ctx_num = 1);

} // namespace inference_core

#endif