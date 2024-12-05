#include "tests/test_func.h"
#include "detection_2d_yolov8/yolov8.h"
#include "detection_2d_rt_detr/rt_detr.h"
#include "image_processing_util/image_processing_util.h"
#include "ort_core/ort_core.h"
#include "tests/fs_util.h"

#include "sam_mobilesam/mobilesam.h"


/**************************
****  ort core test ****
***************************/


TEST(infer_core_test, ort_core_build) 
{
    std::string model_path = "./yolov8n_sim.onnx";
    std::shared_ptr<BaseInferCore> infer_core
                            = CreateOrtInferCore(model_path);
    
    auto map_blob2ptr = infer_core->GetBuffer(false);

    infer_core->SyncInfer(map_blob2ptr);
}


TEST(infer_core_test, ort_core_speed) 
{
    std::string model_path = "./yolov8n_sim.onnx";
    std::shared_ptr<BaseInferCore> infer_core
                            = CreateOrtInferCore(model_path);
    
    test_func_infer_core_speed(infer_core);
}


/***********************************
**** detection with ort test ****
************************************/



TEST(detection_yolov8_test, ort_core_correctness) {
    
    #ifndef PRINT_NO_INFO_LOG
    	LOG(INFO)  << "detection_yolov8_test.ort_core_correctness!";
    #endif
    std::string model_path = "./yolov8n_sim.onnx";
    auto core = CreateOrtInferCore(model_path);
    auto pre_process = CreateCpuDetPreProcess();
    auto post_process = CreateCpuDetPostProcessOrigin(640, 640, 80, {8, 16, 32});
    auto yolov8_det_model = CreateYolov8DetectionModel(core,
                                                        pre_process,
                                                        post_process,
                                                        640,
                                                        640,
                                                        3,
                                                        80,
                                                        {"images"},
                                                        {"output0"});
    
    int res = test_func_yolov8_model_correctness(yolov8_det_model);
    CHECK(res == 11) << ",but got: " << res;
}



TEST(detection_yolov8_test, ort_core_speed) {
    
    #ifndef PRINT_NO_INFO_LOG
    	LOG(INFO)  << "detection_yolov8_test.ort_core_speed!";
    #endif
    std::string model_path = "./yolov8n_sim.onnx";
    auto core = CreateOrtInferCore(model_path);
    auto pre_process = CreateCpuDetPreProcess();
    auto post_process = CreateCpuDetPostProcessOrigin(640, 640, 80, {8, 16, 32});
    auto yolov8_det_model = CreateYolov8DetectionModel(core,
                                                        pre_process,
                                                        post_process,
                                                        640,
                                                        640,
                                                        3,
                                                        80,
                                                        {"images"},
                                                        {"output0"});
    
    int res = test_func_yolov8_model_speed(yolov8_det_model);
    // CHECK(res == 9) << ",but got: " << res;
}




TEST(detection_yolov8_test, ort_core_pipeline_correctness) {
    
    #ifndef PRINT_NO_INFO_LOG
    	LOG(INFO)  << "detection_yolov8_test.ort_core_pipeline_correctness!";
    #endif
    std::string model_path = "./yolov8n_sim.onnx";
    auto core = CreateOrtInferCore(model_path);
    auto pre_process = CreateCpuDetPreProcess();
    auto post_process = CreateCpuDetPostProcessOrigin(640, 640, 80, {8, 16, 32});
    auto yolov8_det_model = CreateYolov8DetectionModel(core,
                                                        pre_process,
                                                        post_process,
                                                        640,
                                                        640,
                                                        3,
                                                        80,
                                                        {"images"},
                                                        {"output0"});
    
    int res = test_func_yolov8_model_pipeline_correctness(yolov8_det_model);
    CHECK(res == 11) << ",but got: " << res;
}




TEST(detection_yolov8_test, ort_core_pipeline_speed) {
    
    #ifndef PRINT_NO_INFO_LOG
    	LOG(INFO)  << "detection_yolov8_test.ort_core_pipeline_correctness!";
    #endif
    std::string model_path = "./yolov8n_sim.onnx";
    auto core = CreateOrtInferCore(model_path);
    auto pre_process = CreateCpuDetPreProcess();
    auto post_process = CreateCpuDetPostProcessOrigin(640, 640, 80, {8, 16, 32});
    auto yolov8_det_model = CreateYolov8DetectionModel(core,
                                                        pre_process,
                                                        post_process,
                                                        640,
                                                        640,
                                                        3,
                                                        80,
                                                        {"images"},
                                                        {"output0"});
    
    int res = test_func_yolov8_model_pipeline_speed(yolov8_det_model);
    // CHECK(res == 9) << ",but got: " << res;
}







TEST(detection_rtdetr_test, ort_core_correctness) {
    
    LOG(INFO)  << "detection_rtdetr_test.ort_core_correctness!";
    std::string model_path = "./rtdetrv2_opset16_single_input.onnx";
    auto core = CreateOrtInferCore(model_path);
    auto pre_process = CreateCpuDetPreProcess();
    auto det_model = CreateRTDetrDetectionModel(core,
                                                pre_process,
                                                640,
                                                640,
                                                3,
                                                80,
                                                {"images"},
                                                {"labels", "boxes", "scores"});
    
    int res = test_func_yolov8_model_correctness(det_model);
}



TEST(detection_rtdetr_test, ort_core_speed) {
    
    LOG(INFO)  << "detection_rtdetr_test.ort_core_speed!";
    std::string model_path = "./rtdetrv2_opset16_single_input.onnx";
    auto core = CreateOrtInferCore(model_path);
    auto pre_process = CreateCpuDetPreProcess();
    auto det_model = CreateRTDetrDetectionModel(core,
                                                pre_process,
                                                640,
                                                640,
                                                3,
                                                80,
                                                {"images"},
                                                {"labels", "boxes", "scores"});
    
    int res = test_func_yolov8_model_speed(det_model);
}


TEST(detection_rtdetr_test, ort_core_pipeline_correctness) {
    
    LOG(INFO)  << "detection_rtdetr_test.ort_core_pipeline_correctness!";
    std::string model_path = "./rtdetrv2_opset16_single_input.onnx";
    auto core = CreateOrtInferCore(model_path);
    auto pre_process = CreateCpuDetPreProcess();
    auto det_model = CreateRTDetrDetectionModel(core,
                                                pre_process,
                                                640,
                                                640,
                                                3,
                                                80,
                                                {"images"},
                                                {"labels", "boxes", "scores"});
    
    int res = test_func_yolov8_model_pipeline_correctness(det_model);
}

TEST(detection_rtdetr_test, ort_core_pipeline_speed) {
    
    LOG(INFO)  << "detection_rtdetr_test.ort_core_pipeline_speed!";
    std::string model_path = "./rtdetrv2_opset16_single_input.onnx";
    auto core = CreateOrtInferCore(model_path);
    auto pre_process = CreateCpuDetPreProcess();
    auto det_model = CreateRTDetrDetectionModel(core,
                                                pre_process,
                                                640,
                                                640,
                                                3,
                                                80,
                                                {"images"},
                                                {"labels", "boxes", "scores"});
    
    int res = test_func_yolov8_model_pipeline_speed(det_model);
}

TEST(sam_mobilesam_test, decoder_ort_core_build)
{
   	LOG(INFO) << "sam_mobilesam_test.decoder_onnx_core_build!";
    std::string model_path = "./mobile_sam_point_single.onnx";
    auto infer_core = CreateOrtInferCore(model_path,
                                         {
                                          {"image_embeddings", {1,256,64,64}},
                                          {"point_coords", {1, 1, 2}},
                                          {"point_labels", {1, 1}},
                                          {"mask_input", {1, 1, 256, 256}},
                                          {"has_mask_input", {1}}
                                         },
                                         {
                                          {"masks", {1, 1, 256, 256}},
                                          {"scores", {1, 1}}
                                         });

    auto map_blob2ptr = infer_core->GetBuffer(false);
    infer_core->SyncInfer(map_blob2ptr);
}



TEST(sam_mobilesam_test, encoder_ort_core_build)
{
   	LOG(INFO) << "sam_mobilesam_test.encoder_onnx_core_build!";
    std::string model_path = "./mobile_sam_encoder_sim.onnx";
    auto infer_core = CreateOrtInferCore(model_path);

    auto map_blob2ptr = infer_core->GetBuffer(false);
    infer_core->SyncInfer(map_blob2ptr);
}






TEST(sam_mobilesam_test, ort_with_point_correctness)
{
   	LOG(INFO) << "sam_mobilesam_test.inference_with_point_correctness!";
    std::string encoder_model_path = "./nanosam_image_encoder_opset11.onnx";
    std::string decoder_model_path = "./modified_mobile_sam_point_single.onnx";
    auto encoder_infer_core = CreateOrtInferCore(encoder_model_path);
    auto decoder_infer_core = CreateOrtInferCore(decoder_model_path,
                                         {
                                          {"image_embeddings", {1,256,64,64}},
                                          {"point_coords", {1, 1, 2}},
                                          {"point_labels", {1, 1}},
                                          {"mask_input", {1, 1, 256, 256}},
                                          {"has_mask_input", {1}}
                                         },
                                         {
                                          {"masks", {1, 1, 256, 256}},
                                          {"scores", {1, 1}}
                                         });

    auto mobile_sam_model = CreateMobileSamModel(encoder_infer_core,
                                    decoder_infer_core,
                                    nullptr,
                                    CreateCpuDetPreProcess());

    test_func_sam_point_correctness(mobile_sam_model);
}


TEST(sam_mobilesam_test, ort_with_point_speed)
{
   	LOG(INFO) << "sam_mobilesam_test.ort_with_point_speed!";
    std::string encoder_model_path = "./nanosam_image_encoder_opset11.onnx";
    std::string decoder_model_path = "./modified_mobile_sam_point_single.onnx";
    auto encoder_infer_core = CreateOrtInferCore(encoder_model_path);
    auto decoder_infer_core = CreateOrtInferCore(decoder_model_path,
                                         {
                                          {"image_embeddings", {1,256,64,64}},
                                          {"point_coords", {1, 1, 2}},
                                          {"point_labels", {1, 1}},
                                          {"mask_input", {1, 1, 256, 256}},
                                          {"has_mask_input", {1}}
                                         },
                                         {
                                          {"masks", {1, 1, 256, 256}},
                                          {"scores", {1, 1}}
                                         });

    auto mobile_sam_model = CreateMobileSamModel(encoder_infer_core,
                                    decoder_infer_core,
                                    nullptr,
                                    CreateCpuDetPreProcess());

    test_func_sam_point_speed(mobile_sam_model);
}


TEST(sam_mobilesam_test, ort_with_box_correctness)
{
   	LOG(INFO) << "sam_mobilesam_test.inference_with_box_correctness!";
    std::string encoder_model_path = "./mobile_sam_encoder_sim.onnx";
    std::string decoder_model_path = "./modified_mobile_sam_box_single.onnx";
    auto encoder_infer_core = CreateOrtInferCore(encoder_model_path);
    auto decoder_infer_core = CreateOrtInferCore(decoder_model_path,
                                         {
                                          {"image_embeddings", {1, 256, 64, 64}},
                                          {"boxes", {1, 1, 4}},
                                          {"mask_input", {1, 1, 256, 256}},
                                          {"has_mask_input", {1}}
                                         },
                                         {
                                          {"masks", {1, 1, 256, 256}},
                                          {"scores", {1, 1}}
                                         });

    auto mobile_sam_model = CreateMobileSamModel(encoder_infer_core,
                                    nullptr,
                                    decoder_infer_core,
                                    CreateCpuDetPreProcess());
    
    test_func_sam_box_correctness(mobile_sam_model);
}



TEST(sam_mobilesam_test, ort_with_point_pipeline_correctness)
{
   	LOG(INFO) << "sam_mobilesam_test.ort_with_point_pipeline_correctness!";
    std::string encoder_model_path = "./nanosam_image_encoder_opset11.onnx";
    std::string decoder_model_path = "./modified_mobile_sam_point_single.onnx";
    auto encoder_infer_core = CreateOrtInferCore(encoder_model_path);
    auto decoder_infer_core = CreateOrtInferCore(decoder_model_path,
                                         {
                                          {"image_embeddings", {1,256,64,64}},
                                          {"point_coords", {1, 1, 2}},
                                          {"point_labels", {1, 1}},
                                          {"mask_input", {1, 1, 256, 256}},
                                          {"has_mask_input", {1}}
                                         },
                                         {
                                          {"masks", {1, 1, 256, 256}},
                                          {"scores", {1, 1}}
                                         });

    auto mobile_sam_model = CreateMobileSamModel(encoder_infer_core,
                                    decoder_infer_core,
                                    nullptr,
                                    CreateCpuDetPreProcess());

    test_func_sam_point_pipeline_correctness(mobile_sam_model);
}


TEST(sam_mobilesam_test, ort_with_point_pipeline_speed)
{
   	LOG(INFO) << "sam_mobilesam_test.ort_with_point_pipeline_speed!";
    std::string encoder_model_path = "./nanosam_image_encoder_opset11.onnx";
    std::string decoder_model_path = "./modified_mobile_sam_point_single.onnx";
    auto encoder_infer_core = CreateOrtInferCore(encoder_model_path);
    auto decoder_infer_core = CreateOrtInferCore(decoder_model_path,
                                         {
                                          {"image_embeddings", {1,256,64,64}},
                                          {"point_coords", {1, 1, 2}},
                                          {"point_labels", {1, 1}},
                                          {"mask_input", {1, 1, 256, 256}},
                                          {"has_mask_input", {1}}
                                         },
                                         {
                                          {"masks", {1, 1, 256, 256}},
                                          {"scores", {1, 1}}
                                         });

    auto mobile_sam_model = CreateMobileSamModel(encoder_infer_core,
                                    decoder_infer_core,
                                    nullptr,
                                    CreateCpuDetPreProcess());

    test_func_sam_point_pipeline_speed(mobile_sam_model);
}

TEST(sam_mobilesam_test, ort_with_box_pipeline_correctness)
{
   	LOG(INFO) << "sam_mobilesam_test.ort_with_box_pipeline_correctness!";
    std::string encoder_model_path = "./mobile_sam_encoder_sim.onnx";
    std::string decoder_model_path = "./modified_mobile_sam_box_single.onnx";
    auto encoder_infer_core = CreateOrtInferCore(encoder_model_path);
    auto decoder_infer_core = CreateOrtInferCore(decoder_model_path,
                                         {
                                          {"image_embeddings", {1, 256, 64, 64}},
                                          {"boxes", {1, 1, 4}},
                                          {"mask_input", {1, 1, 256, 256}},
                                          {"has_mask_input", {1}}
                                         },
                                         {
                                          {"masks", {1, 1, 256, 256}},
                                          {"scores", {1, 1}}
                                         });

    auto mobile_sam_model = CreateMobileSamModel(encoder_infer_core,
                                    nullptr,
                                    decoder_infer_core,
                                    CreateCpuDetPreProcess());
    
    test_func_sam_box_pipeline_correctness(mobile_sam_model);
}





TEST(sam_mobilesam_test, ort_with_point_dynamic_correctness)
{
   	LOG(INFO) << "sam_mobilesam_test.trt_with_point_dynamic_correctness!";
    std::string encoder_model_path = "./mobile_sam_encoder_sim.onnx";
    std::string decoder_model_path = "./mobile_sam_point_single.onnx";
    auto encoder_infer_core = CreateOrtInferCore(encoder_model_path);
    auto decoder_infer_core = CreateOrtInferCore(decoder_model_path,
                                         {
                                          {"image_embeddings", {1,256,64,64}},
                                          {"point_coords", {1, 8, 2}},
                                          {"point_labels", {1, 8}},
                                          {"mask_input", {1, 1, 256, 256}},
                                          {"has_mask_input", {1}}
                                         },
                                         {
                                          {"masks", {1, 1, 256, 256}},
                                          {"scores", {1, 1}}
                                         });

    auto model = CreateMobileSamModel(encoder_infer_core,
                                    decoder_infer_core,
                                    nullptr,
                                    CreateCpuDetPreProcess());
    
    auto image = cv::imread("./persons.jpg");

    std::vector<std::pair<int,int>> points {
        {225, 370},
        {350, 400},
        {550, 300}
    };
    std::vector<int> labels {
        0,
        0,
        1
    };

    ImageDrawHelper helper(std::make_shared<cv::Mat>(image.clone()));
    for (const auto& point : points) {
        helper.drawPoint(point);
    }

    cv::Mat masks = model->GenerateMask(image, points, labels, false);
    cv::imshow("masks", masks);
    cv::imshow("points", *helper.getImage());
    cv::waitKey(0);
}



