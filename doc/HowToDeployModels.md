# How to Deploy Models with `EasyDeploy`

We designed `BaseInferCore` and `IBlobsBuffer` to unify inference across frameworks, handling memory transport and enabling easy read/write without framework or hardware concerns.  

This allows running multiple inference cores (e.g., `onnxruntime` and `rknn` on `RK3588`) to maximize throughput. The async pipeline further boosts performance.  

Key details and virtual functions are in `base_infer_core.h`; developers only implement data processing and `blob_buffer` allocation.

## `inference_core::BaseInferCore`

When using BaseInferCore, algorithm developers only need to:
1. Implement the key abstract functions defined by BaseInferCore.
2. Use the IBlobsBuffer interface for managing inference buffers, without worrying about the underlying buffer logic.
3. Focus on the deployment of the model inference process while leaving the specifics of the inference framework and algorithm type abstracted away.
    
- Example:
    ```cpp
    auto infer_core = std::make_shared<TrtInferCore>(...);
    auto blob_buffer = infer_core->GetBuffer(true); // get blob buffer from pool
    auto input_buffer_and_loc = blob_buffer->GetOuterBlobBuffer("images");
    // input_buffer_and_loc contains buffer ptr and location
    // the location is on host-side by default
    void* input_buffer_ptr = input_buffer_and_loc.first;
    memcpy(input_buffer_ptr, src, len);
    // do inference
    infer_core->SyncInfer(blob_buffer);
    // postprocess
    auto output_buffer_and_loc = blob_buffer->GetOuterBlobBuffer("output");
    void* output_buffer_ptr = output_buffer_and_loc.first;
    ...
    ```

- Through the `IBlobsBuffer` interface, **dynamic shape** of a specific blob can be configured, or an existing buffer can be reused, enabling zero-copy in sequential model inference workflows.  

- Some inference frameworks or hardware platforms are heterogeneous, requiring data transfers between the host and device (e.g., platforms related to the TensorRT framework). These tasks are handled in the concrete implementation classes of `BaseInferCore`. During usage, users can select host or device-side buffers via the `SetBlobBuffer` method. The buffers retrieved through the `GetOuterBlobBuffer` method will adapt accordingly.  

- The overloaded `SetBlobBuffer` method allows users to specify external buffers, enabling zero-copy functionality.