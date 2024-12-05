#!/bin/bash
echo "Converting yolov8 ..."
/usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/yolov8n.onnx \
                              --saveEngine=/workspace/models/yolov8n.engine \
                              --fp16

echo "Converting rt_detr_v2 ..."
/usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/rt_detr_v2_single_input.onnx \
                              --saveEngine=/workspace/models/rt_detr_v2_single_input.engine

echo "Converting mobilesam ..."
/usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/mobile_sam_encoder.onnx \
                              --saveEngine=/workspace/models/mobile_sam_encoder.engine

echo "Converting nanosam ..."
/usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/nanosam_image_encoder_opset11.onnx \
                              --saveEngine=/workspace/models/nanosam_image_encoder_opset11.engine \
                              --fp16

echo "Converting mobilesam_box_decoder ..."
/usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/modified_mobile_sam_box.onnx \
                              --saveEngine=/workspace/models/modified_mobile_sam_box.engine \
                              --fp16

echo "Converting mobilesam_point_decoder ..."
/usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/modified_mobile_sam_point.onnx \
                              --saveEngine=/workspace/models/modified_mobile_sam_point.engine \
                              --fp16
