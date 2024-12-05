#!/bin/bash
echo "Converting yolov8 ..."
python3 /workspace/tools/cvt_onnx2rknn_yolov8_quant.py

echo "Converting nanosam ..."
python3 /workspace/tools/cvt_onnx2rknn_nanosam.py

echo "Converting mobilesam_box_decoder ..."
python3 /workspace/tools/cvt_onnx2rknn_mobilesam_box_decoder.py

echo "Converting mobilesam_point_decoder ..."
python3 /workspace/tools/cvt_onnx2rknn_mobilesam_point_decoder.py