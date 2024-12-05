/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-19 18:33:00
 * @LastEditTime: 2024-12-03 15:51:37
 * @FilePath: /EasyDeploy/deploy_utils/image_processing_utils/include/detection_2d_util/detection_2d_util.h
 */
#ifndef __EASY_DEPLOY_DETECTION_2D_UTIL_H
#define __EASY_DEPLOY_DETECTION_2D_UTIL_H

#include "deploy_core/base_detection.h"
#include "deploy_core/base_infer_core.h"

namespace detection_2d {

/**
 * @brief Create a Cpu based Det Pre Process object
 *
 */
std::shared_ptr<IDetectionPreProcess> CreateCpuDetPreProcess(
    const std::vector<float> &mean         = {0, 0, 0},
    const std::vector<float> &val          = {255, 255, 255},
    bool                      do_transpose = true,
    bool                      do_norm      = true);

std::shared_ptr<BaseDetectionPreprocessFactory> CreateCpuDetPreProcessFactory(
    const std::vector<float> &mean         = {0, 0, 0},
    const std::vector<float> &val          = {255, 255, 255},
    bool                      do_transpose = true,
    bool                      do_norm      = true);

/**
 * @brief Create a Cuda based Det Pre Process object
 *
 */
std::shared_ptr<IDetectionPreProcess> CreateCudaDetPreProcess(const int max_src_height   = 1920,
                                                              const int max_src_width    = 1920,
                                                              const int max_src_channels = 3);

std::shared_ptr<BaseDetectionPreprocessFactory> CreateCudaDetPreProcessFactory(
    const int max_src_height   = 1920,
    const int max_src_width    = 1920,
    const int max_src_channels = 3);

/**
 * @brief Refer to `ultralytics` official project.
 *
 */
std::shared_ptr<IDetectionPostProcess> CreateYolov8PostProcessCpuOrigin(
    const int               input_height,
    const int               input_width,
    const int               cls_number,
    const std::vector<int> &downsample_scales = {8, 16, 32});

std::shared_ptr<BaseDetectionPostprocessFactory> CreateYolov8PostProcessCpuOriginFactory(
    const int               input_height,
    const int               input_width,
    const int               cls_number,
    const std::vector<int> &downsample_scales = {8, 16, 32});

/**
 * @brief Refer to `ultralytics` official project.
 *
 * In ultralytics v8.2.2, original yolov8 output shape: `[batch, 4 + cls_number, 8400]`.
 * We Modify output shape to `[batch, 8400, 4 + cls_number]`
 *
 */
std::shared_ptr<IDetectionPostProcess> CreateYolov8PostProcessCpuTranspose(
    const int               input_height,
    const int               input_width,
    const int               cls_number,
    const std::vector<int> &downsample_scales = {8, 16, 32});

std::shared_ptr<BaseDetectionPostprocessFactory> CreateYolov8PostProcessCpuTransposeFactory(
    const int               input_height,
    const int               input_width,
    const int               cls_number,
    const std::vector<int> &downsample_scales = {8, 16, 32});

/**
 * @brief refer to modified `ultralytics` project by rk team。
 * @brief <rknn example> `https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8`
 * @brief <erase `drl` module>
 * `https://github.com/airockchip/ultralytics_yolov8/blob/main/RKOPT_README.md`
 *
 */
std::shared_ptr<IDetectionPostProcess> CreateYolov8PostProcessCpuDivide(
    const int               input_height,
    const int               input_width,
    const int               cls_number,
    const std::vector<int> &downsample_scales = {8, 16, 32});

std::shared_ptr<BaseDetectionPostprocessFactory> CreateYolov8PostProcessCpuDivideFactory(
    const int               input_height,
    const int               input_width,
    const int               cls_number,
    const std::vector<int> &downsample_scales = {8, 16, 32});

} // namespace detection_2d

#endif