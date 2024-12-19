# EasyDeploy Quick Start Demo

This documentation will show how to use `EasyDeploy` with the `yolov8`, `rt-detr`, `mobile-sam`, `nano-sam` algorithms on multiple inference frameworks.

Before this, you should follow [Setup](./EnviromentSetup.md)

## 0. QuickStart

### 0.1 Download all onnx model
  - All `onnx` models could be downloaded from [goolge driver](https://drive.google.com/drive/folders/1yVEOzo59aob_1uXwv343oeh0dTKuHT58?usp=drive_link).
  - Put all onnx models into `EasyDeploy/models` directory.

### 0.2 Convert models
  - Use `tools/cvt_onnx2*_all.sh` scripts to convert onnx models.
  ```bash
  bash tools/cvt_onnx2rknn_all.sh
  # bash tools/cvt_onnx2trt_all.sh
  ```
### 0.3 Run demo
  - Build and run simple_tests demo. See test cases for detail.
  ```bash
    cd /workspace
    mkdir build && cd build
    cmake .. -DBUILD_TESTING=ON -DENABLE_TENSORRT=ON
    make -j

    # test on yolo correctness
    ./bin/simple_tests --gtest_filter=*yolo*correctness
    # test on yolo speed
    GLOG_minloglevel=1 ./bin/simple_tests --gtest_filter=*yolo*speed
    # for other tests, please see `EasyDeploy/simple_tests/src/`
  ```



## 1. Yolov8

### 1.1 Yolov8-with-TensorRT

#### 1.1.1 Export yolov8 onnx model from `ultralytics` project.

  - Install `ultralytics` by pip
    ```bash
    pip install ultralytics
    ```

  - Export Yolov8n model to onnx, and copy it to `path/to/EasyDeploy/models`.
    ```bash
    # download the official model online
    yolo export model=yolov8n.pt format=onnx
    # use the local pre-downloaded model or your custom model
    yolo export model=./yolov8n.pt format=onnx
    ```

#### 1.1.2 Build TensorRT engine with the yolov8 onnx model.

  - Go into your pre-built tensorrt docker container.
    ```bash
    cd path/to/EasyDeploy/
    bash docker/into_docker.sh
    ```
  
  - Use `trtexec` official tool to convert onnx model to tensorrt engine.
    ```bash
    /usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/yolov8n.onnx \
                                  --saveEngine=/workspace/models/yolov8n.engine \
                                  --fp16
    ```

#### 1.1.3 Run test demo
  - Now we have a `yolov8n.engine` file under `/workspace/models/`. Run simple tests on yolov8.
    ```bash
    cd /workspace
    mkdir build && cd build
    cmake .. -DBUILD_TESTING=ON -DENABLE_TENSORRT=ON
    make -j

    # test on yolo correctness
    ./bin/simple_tests --gtest_filter=*yolo*correctness
    # test on yolo speed
    GLOG_minloglevel=1 ./bin/simple_tests --gtest_filter=*yolo*speed
    ```


### 1.2 Yolov8-with-RKNN






## 2. RT-Detr

### 2.1 RT-Detr-with-TensorRT

#### 2.1.1 Export RT-Detr-V2 onnx model
  - Reference to official [RT-Detr-v2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch). The original model exported by `tools/export_onnx.py` supports dynamic input shape. However, we simply want a static input shape model for inference. So, the script we used to export the onnx model with single static input image blob is `EasyDeploy/tools/rt_detr_v2_export_onnx.py`. Replace the official export script with our script.
    - Download the official rt-detr code from [rt-detr](https://github.com/lyuwenyu/RT-DETR/tree/main).
      ```bash
      git clone git@github.com:lyuwenyu/RT-DETR.git
      ```

    - Setup enviroment and download models
      ```bash
      cd rtdetrv2_pytorch/
      pip install -r requirements.txt

      mkdir weights && cd weights
      wget https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
      ```

    - Export rt-detr-v2 model with single static input image blob.
      ```bash
      cp EasyDeploy/tools/rt_detr_v2_export_onnx.py RT-DETR/rtdetrv2/tools/
      cd RT-DETR/rtdetrv2

      python3 tools/rt_detr_v2_export_onnx.py \
              -c ./configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
              -r weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
              -o rt_detr_v2_single_input.onnx --check --simplify
      ```

    - Visualize the output onnx model with [Netron](https://netron.app/). The image input blob should be static shape and the image original shape blob should be removed.

#### 2.1.2 Build TensorRT engine with rt-detr-v2 onnx model
  - Now we get the rt-detr-v2 onnx model. Convert it to tensorrt engine with `trtexec` tool. Note that `fp16` optimization will greatly affect the model's accuracy. So, we do not convert the onnx model with `--fp16` flag.
    ```bash
    /usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/rt_detr_v2_single_input.onnx \
                                  --saveEngine=/workspace/models/rt_detr_v2_single_input.engine
    ```

#### 2.1.3 Run test demo
  - Then we get the rt-detr-v2 tensorrt engine model. Run simple tests.
    ```bash
    cd /workspace
    mkdir build && cd build
    cmake .. -DBUILD_TESTING=ON -DENABLE_TENSORRT=ON
    make -j

    # test on rt-detr-v2 correctness
    ./bin/simple_tests --gtest_filter=*rtdetr*correctness
    # test on rt-detr-v2 speed
    GLOG_minloglevel=1 ./bin/simple_tests --gtest_filter=*rtdetr*speed
    ```


## 3. MobileSAM and NanoSAM

### 3.1 Mobile/Nano-SAM-with-TensorRT

#### 3.1.1 Get the onnx model

  - Download the onnx models from [goolge driver](https://drive.google.com/drive/folders/1yVEOzo59aob_1uXwv343oeh0dTKuHT58?usp=drive_link). Put models under `EasyDeploy/models/`.

#### 3.1.2 Build TensorRT engine with onnx model
  - Convert models with `trtexec` tool.
    ```bash
    bash docker/into_docker.sh

    # convert mobile-sam image encoder
    /usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/mobile_sam_encoder.onnx \
                                  --saveEngine=/workspace/models/mobile_sam_encoder.engine
    # convert nano-sam image encoder
    /usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/nanosam_image_encoder_opset11.onnx \
                                  --saveEngine=/workspace/models/nanosam_image_encoder_opset11.engine \
                                  --fp16
    # convert decoder-with-point
    /usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/modified_mobile_sam_point.onnx \
                                  --saveEngine=/workspace/models/modified_mobile_sam_point.engine \
                                  --fp16
    # convert decoder-with-box
    /usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/modified_mobile_sam_box.onnx \
                                  --saveEngine=/workspace/models/modified_mobile_sam_box.engine \
                                  --fp16                   
    ```

#### 3.1.3 Run test demo
  - Now we get all tensorrt engine models used in mobilesam/nanosam. Run simple tests.
    ```bash
    cd /workspace
    mkdir build && cd build
    cmake .. -DBUILD_TESTING=ON -DENABLE_TENSORRT=ON
    make -j

    # test on mobilesam correctness
    ./bin/simple_tests --gtest_filter=*mobilesam*correctness
    # test on mobilesam speed
    GLOG_minloglevel=1 ./bin/simple_tests --gtest_filter=*mobilesam*speed
    ```

  - **Note** You may want to change the image encoder path in `/workspace/simple_tests/src/test_jetson_devkit.cpp` to `/workspace/models/nanosam_image_encoder_opset11.engine`, which could turn the mobilesam algorithm into nanosam algorithm.

### 3.2 Nano-SAM-with-RKNN

#### 3.2.1 Get the onnx model

  - Download the onnx models from [goolge driver](https://drive.google.com/drive/folders/1yVEOzo59aob_1uXwv343oeh0dTKuHT58?usp=drive_link). Put models under `EasyDeploy/models/`.

#### 3.2.2 Build rknn model from onnx model

  - Use python scripts in `EasyDeploy/tools/`. The python enviroment should been setup in docker already. 
    ```bash
    cd tools
    python3 cvt_onnx2rknn_nanosam.py
    python3 cvt_onnx2rknn_mobilesam_point_decoder.py
    python3 cvt_onnx2rknn_mobilesam_box_decoder.py
    ```

#### 3.2.3 Run test demo
  - Run demos under `simple_tests/src/test_rk_devkit.cpp`.
    ```bash
    cd /workspace
    mkdir build && cd build
    cmake .. -DBUILD_TESTING=ON -DENABLE_RKNN=ON
    make -j

    # test on mobilesam correctness
    ./bin/simple_tests --gtest_filter=*mobilesam*correctness
    # test on mobilesam speed
    GLOG_minloglevel=1 ./bin/simple_tests --gtest_filter=*mobilesam*speed
    ```