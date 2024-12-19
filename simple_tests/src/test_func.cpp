#include "tests/test_func.h"
#include "deploy_core/wrapper.h"

float test_func_infer_core_speed(std::shared_ptr<BaseInferCore> core)
{
  auto       map_blob2ptr = core->AllocBlobsBuffer();
  FPSCounter fps_counter;
  fps_counter.Start();
  for (int i = 0; i < 500; ++i)
  {
    core->SyncInfer(map_blob2ptr);
    fps_counter.Count(1);
  }
  LOG(WARNING) << "average fps: " << fps_counter.GetFPS();
  return fps_counter.GetFPS();
}

float test_func_yolov8_model_speed(std::shared_ptr<BaseDetectionModel> model,
                                   std::string                         test_image_path)
{
  cv::Mat    fake_image = cv::imread(test_image_path);
  FPSCounter fps_counter;
  fps_counter.Start();
  for (int i = 0; i < 500; ++i)
  {
    std::vector<BBox2D> results;
    model->Detect(fake_image.clone(), results, 0.4);
    fps_counter.Count(1);
  }
  LOG(WARNING) << "average fps: " << fps_counter.GetFPS();
  return fps_counter.GetFPS();
}

int test_func_yolov8_model_correctness(std::shared_ptr<BaseDetectionModel> model,
                                       std::string                         test_image_path,
                                       std::string                         test_results_save_path)
{
  cv::Mat             fake_image = cv::imread(test_image_path);
  std::vector<BBox2D> results;
  auto                start = std::chrono::high_resolution_clock::now();
  model->Detect(fake_image, results, 0.4);
  auto end = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "do_inference, cost : "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  ImageDrawHelper drawer(std::make_shared<cv::Mat>(fake_image.clone()));
  for (const auto &res : results)
  {
    drawer.drawRect2D(res);
  }
  auto p_image = drawer.getImage();
  cv::imwrite(test_results_save_path, *p_image);

  LOG(WARNING) << "remain objects number: " << results.size();
  return results.size();
}

float test_func_yolov8_model_pipeline_speed(std::shared_ptr<BaseDetectionModel> model,
                                            std::string                         test_image_path)
{
  model->InitPipeline();
  cv::Mat fake_image = cv::imread(test_image_path);

  BlockQueue<std::shared_ptr<std::future<std::vector<BBox2D>>>> future_bq(100);

  auto func_push_data = [&]() {
    int index = 0;
    while (index++ < 2000)
    {
      auto p_fut = std::make_shared<std::future<std::vector<BBox2D>>>(
          model->DetectAsync(fake_image.clone(), 0.4));
      future_bq.BlockPush(p_fut);
    }
    future_bq.SetNoMoreInput();
  };

  FPSCounter fps_counter;
  auto       func_take_results = [&]() {
    int index = 0;
    fps_counter.Start();
    while (true)
    {
      auto output = future_bq.Take();
      if (!output.has_value())
        break;
      output.value()->get();
      fps_counter.Count(1);
    }
  };

  std::thread t_push(func_push_data);
  std::thread t_take(func_take_results);

  t_push.join();
  model->StopPipeline();
  t_take.join();
  model->ClosePipeline();

  LOG(WARNING) << "average fps: " << fps_counter.GetFPS();
  return fps_counter.GetFPS();
}

int test_func_yolov8_model_pipeline_correctness(std::shared_ptr<BaseDetectionModel> model,
                                                std::string                         test_image_path,
                                                std::string test_results_save_path)
{
  model->InitPipeline();

  cv::Mat fake_image = cv::imread(test_image_path);

  auto future = model->DetectAsync(fake_image, 0.4);

  std::vector<BBox2D> results = future.get();

  ImageDrawHelper drawer(std::make_shared<cv::Mat>(fake_image.clone()));
  for (const auto &res : results)
  {
    drawer.drawRect2D(res);
  }
  auto p_image = drawer.getImage();
  cv::imwrite(test_results_save_path, *p_image);

  model->StopPipeline();
  model->ClosePipeline();
  LOG(WARNING) << "remain objects number: " << results.size();
  return results.size();
}

// int test_func_yolov8_model_pipeline_correctness_callback(std::shared_ptr<BaseDetectionModel>
// model,
//                                             std::string test_image_path,
//                                             std::string test_results_save_path)
// {
//     model->InitPipeline();

//     bool flag = false;
//     int detected_obj_num = 0;

//     cv::Mat fake_image = cv::imread(test_image_path);

//     model->DetectAsync(fake_image,
//                     0.4,
//                     [&](const std::vector<BBox2D>& results, int index) {
//                         ImageDrawHelper drawer(std::make_shared<cv::Mat>(fake_image.clone()));
//                         for (const auto & res : results) {
//                             drawer.drawRect2D(res);
//                         }
//                         auto p_image = drawer.getImage();
//                         cv::imwrite(test_results_save_path, *p_image);
//                         LOG(WARNING) << "remain objects number: " << results.size();
//                         detected_obj_num = results.size();
//                         flag = true;
//                     });

//     while (flag == false) {}

//     model->StopPipeline();
//     model->ClosePipeline();
//     return detected_obj_num;
// }

void test_func_sam_point_correctness(std::shared_ptr<BaseSamModel> model,
                                     std::string                   test_image_path,
                                     std::string                   test_results_save_path)
{
  cv::Mat image_test = cv::imread(test_image_path);
  cv::Mat masks;
  model->GenerateMask(image_test, {{225, 370}}, std::vector<int>{1}, masks, false);

  ImageDrawHelper helper(std::make_shared<cv::Mat>(image_test.clone()));
  helper.addRedMaskToForeground(masks);

  cv::imwrite(test_results_save_path, *helper.getImage());
}

void test_func_sam_box_correctness(std::shared_ptr<BaseSamModel> model,
                                   std::string                   test_image_path,
                                   std::string                   test_results_save_path)
{
  cv::Mat image_test = cv::imread(test_image_path);
  cv::Mat masks;
  model->GenerateMask(image_test, {{225, 370, 110, 300}}, masks);

  ImageDrawHelper helper(std::make_shared<cv::Mat>(image_test.clone()));
  helper.addRedMaskToForeground(masks);

  cv::imwrite(test_results_save_path, *helper.getImage());
}

void test_func_sam_point_pipeline_correctness(std::shared_ptr<BaseSamModel> model,
                                              std::string                   test_image_path,
                                              std::string                   test_results_save_path)
{
  cv::Mat image_test = cv::imread(test_image_path);

  model->InitPipeline();

  for (int i = 0; i < 5; ++i)
  {
    auto fut = model->GenerateMaskAsync(image_test, {{225, 370}}, std::vector<int>{1});

    cv::Mat masks = fut.get();
    ImageDrawHelper helper(std::make_shared<cv::Mat>(image_test.clone()));
    helper.addRedMaskToForeground(masks);

    cv::imwrite("/workspace/test_data/tests_masks_output_" + std::to_string(i) + ".png", *helper.getImage());
  }
}

void test_func_sam_box_pipeline_correctness(std::shared_ptr<BaseSamModel> model,
                                            std::string                   test_image_path,
                                            std::string                   test_results_save_path)
{
  cv::Mat image_test = cv::imread(test_image_path);

  model->InitPipeline();

  for (int i = 0; i < 5; ++i)
  {
    auto fut = model->GenerateMaskAsync(image_test, {{225, 370, 110, 300}});

    cv::Mat masks = fut.get();
    ImageDrawHelper helper(std::make_shared<cv::Mat>(image_test.clone()));
    helper.addRedMaskToForeground(masks);

    cv::imwrite("/workspace/test_data/tests_masks_output_" + std::to_string(i) + ".png", *helper.getImage());
  }
}

float test_func_sam_point_speed(std::shared_ptr<BaseSamModel> model,
                                std::string                   test_image_path,
                                std::string                   test_results_save_path)
{
  FPSCounter fps_counter;
  fps_counter.Start();
  cv::Mat image_test = cv::imread(test_image_path);
  for (int i = 0; i < 100; ++i)
  {
    cv::Mat masks;
    model->GenerateMask(image_test, {{225, 370}}, {1}, masks, false);
    fps_counter.Count(1);
  }
  LOG(WARNING) << "average fps: " << fps_counter.GetFPS();

  return fps_counter.GetFPS();
}

float test_func_sam_box_speed(std::shared_ptr<BaseSamModel> model,
                              std::string                   test_image_path,
                              std::string                   test_results_save_path)
{
  FPSCounter fps_counter;
  fps_counter.Start();
  cv::Mat image_test = cv::imread(test_image_path);
  for (int i = 0; i < 100; ++i)
  {
    cv::Mat masks;
    model->GenerateMask(image_test, {{225, 370, 110, 300}}, masks, false);
    fps_counter.Count(1);
  }
  LOG(WARNING) << "average fps: " << fps_counter.GetFPS();

  return fps_counter.GetFPS();
}

float test_func_sam_point_pipeline_speed(std::shared_ptr<BaseSamModel> model,
                                         std::string                   test_image_path,
                                         std::string                   test_results_save_path)
{
  cv::Mat image_test = cv::imread(test_image_path);

  model->InitPipeline();

  BlockQueue<std::shared_ptr<std::future<cv::Mat>>> future_bq(100);

  auto func_push_data = [&]() {
    int index = 0;
    while (index++ < 200)
    {
      future_bq.BlockPush(std::make_shared<std::future<cv::Mat>>(
          model->GenerateMaskAsync(image_test, {{225, 370}}, std::vector<int>{1})));
    }
    future_bq.SetNoMoreInput();
  };

  FPSCounter fps_counter;
  auto       func_take_results = [&]() {
    int index = 0;
    fps_counter.Start();
    while (true)
    {
      auto res = future_bq.Take();
      if (!res.has_value())
        break;
      res.value()->get();
      fps_counter.Count(1);
    }
  };

  std::thread t_push(func_push_data);
  std::thread t_take(func_take_results);

  t_push.join();
  model->StopPipeline();
  t_take.join();
  model->ClosePipeline();

  LOG(WARNING) << "average fps: " << fps_counter.GetFPS();
  return fps_counter.GetFPS();
}

float test_func_sam_box_pipeline_speed(std::shared_ptr<BaseSamModel> model,
                                       std::string                   test_image_path,
                                       std::string                   test_results_save_path)
{
  cv::Mat image_test = cv::imread(test_image_path);

  model->InitPipeline();

  BlockQueue<std::shared_ptr<std::future<cv::Mat>>> future_bq(100);

  auto func_push_data = [&]() {
    int index = 0;
    while (index++ < 200)
    {
      future_bq.BlockPush(std::make_shared<std::future<cv::Mat>>(
          model->GenerateMaskAsync(image_test, {{225, 370, 110, 300}})));
    }
    future_bq.SetNoMoreInput();
  };

  FPSCounter fps_counter;
  auto       func_take_results = [&]() {
    int index = 0;
    fps_counter.Start();
    while (true)
    {
      auto res = future_bq.Take();
      if (!res.has_value())
        break;
      res.value()->get();
      fps_counter.Count(1);
    }
  };

  std::thread t_push(func_push_data);
  std::thread t_take(func_take_results);

  t_push.join();
  model->StopPipeline();
  t_take.join();
  model->ClosePipeline();

  LOG(WARNING) << "average fps: " << fps_counter.GetFPS();
  return fps_counter.GetFPS();
}
