import sys
from rknn.api import RKNN
# onnx model path
onnx_model_path = '/workspace/models/modified_mobile_sam_point.onnx'
# quant data
DATASET_PATH = '/workspace/test_data/quant_data/dataset.txt'
# output path
DEFAULT_RKNN_PATH = '/workspace/models/modified_mobile_sam_point.rknn'
if __name__ == '__main__':
    # Create RKNN object
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588",
                optimization_level=2)

    ret = rknn.load_onnx(model=onnx_model_path,
                         inputs=['image_embeddings', 'point_coords', 'point_labels', 'mask_input', 'has_mask_input'],
                         input_size_list=[[1,256,64,64], [1,1,2], [1,1], [1,1,256,256], [1]])
    ret = rknn.build(do_quantization=False)

    ret = rknn.export_rknn(DEFAULT_RKNN_PATH)
    print('done')

    # Release
    rknn.release()
