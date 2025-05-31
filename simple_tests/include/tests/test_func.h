#ifndef __TESTS_ALL_IN_ONE_TEST_FUNC_H
#define __TESTS_ALL_IN_ONE_TEST_FUNC_H

#include "deploy_core/base_detection.h"
#include "deploy_core/base_sam.h"
#include "tests/image_drawer.h"
#include "tests/fps_counter.h"
#include "tests/fs_util.h"

#include <glog/logging.h>
#include <glog/log_severity.h>
#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace inference_core;

using namespace detection_2d;
using namespace async_pipeline;
using namespace sam;

class DumbInputImageData : public IPipelineImageData {
public:
  DumbInputImageData(const cv::Mat &cv_image) : inner_cv_image(cv_image)
  {
    image_data_info.data_pointer   = cv_image.data;
    image_data_info.format         = ImageDataFormat::BGR;
    image_data_info.image_height   = cv_image.rows;
    image_data_info.image_width    = cv_image.cols;
    image_data_info.image_channels = cv_image.channels();
    image_data_info.location       = DataLocation::HOST;
  }

  const ImageDataInfo &GetImageDataInfo() const
  {
    return image_data_info;
  }

private:
  IPipelineImageData::ImageDataInfo image_data_info;
  cv::Mat                           inner_cv_image;
};

float test_func_infer_core_speed(std::shared_ptr<BaseInferCore> core);

float test_func_yolov8_model_speed(
    std::shared_ptr<BaseDetectionModel> model,
    std::string                         test_image_path = "/workspace/test_data/persons.jpg");

int test_func_yolov8_model_correctness(
    std::shared_ptr<BaseDetectionModel> model,
    std::string                         test_image_path = "/workspace/test_data/persons.jpg",
    std::string test_results_save_path = "/workspace/test_data/test_persons_detection_results.jpg");

float test_func_yolov8_model_pipeline_speed(
    std::shared_ptr<BaseDetectionModel> model,
    std::string                         test_image_path = "/workspace/test_data/persons.jpg");

int test_func_yolov8_model_pipeline_correctness(
    std::shared_ptr<BaseDetectionModel> model,
    std::string                         test_image_path = "/workspace/test_data/persons.jpg",
    std::string test_results_save_path = "/workspace/test_data/test_persons_detection_results.jpg");

// int test_func_yolov8_model_pipeline_correctness_callback(std::shared_ptr<BaseDetectionModel>
// model,
//                                     std::string test_image_path =
//                                     "/workspace/test_data/persons.jpg", std::string
//                                     test_results_save_path =
//                                     "/workspace/test_data/test_persons_detection_results.jpg");

void test_func_sam_point_correctness(
    std::shared_ptr<BaseSamModel> model,
    std::string                   test_image_path = "/workspace/test_data/persons.jpg",
    std::string test_results_save_path            = "/workspace/test_data/tests_masks_output.png");

void test_func_sam_box_correctness(
    std::shared_ptr<BaseSamModel> model,
    std::string                   test_image_path = "/workspace/test_data/persons.jpg",
    std::string test_results_save_path            = "/workspace/test_data/tests_masks_output.png");

void test_func_sam_point_pipeline_correctness(
    std::shared_ptr<BaseSamModel> model,
    std::string                   test_image_path = "/workspace/test_data/persons.jpg",
    std::string test_results_save_path            = "/workspace/test_data/tests_masks_output.png");

void test_func_sam_box_pipeline_correctness(
    std::shared_ptr<BaseSamModel> model,
    std::string                   test_image_path = "/workspace/test_data/persons.jpg",
    std::string test_results_save_path            = "/workspace/test_data/tests_masks_output.png");

float test_func_sam_point_speed(
    std::shared_ptr<BaseSamModel> model,
    std::string                   test_image_path = "/workspace/test_data/persons.jpg",
    std::string test_results_save_path            = "/workspace/test_data/tests_masks_output.png");

float test_func_sam_box_speed(
    std::shared_ptr<BaseSamModel> model,
    std::string                   test_image_path = "/workspace/test_data/persons.jpg",
    std::string test_results_save_path            = "/workspace/test_data/tests_masks_output.png");

float test_func_sam_point_pipeline_speed(
    std::shared_ptr<BaseSamModel> model,
    std::string                   test_image_path = "/workspace/test_data/persons.jpg",
    std::string test_results_save_path            = "/workspace/test_data/tests_masks_output.png");

float test_func_sam_box_pipeline_speed(
    std::shared_ptr<BaseSamModel> model,
    std::string                   test_image_path = "/workspace/test_data/persons.jpg",
    std::string test_results_save_path            = "/workspace/test_data/tests_masks_output.png");

#endif
