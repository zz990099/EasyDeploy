#include "ort_core/ort_core.h"

#include <onnxruntime_cxx_api.h>
#include <glog/logging.h>
#include <glog/log_severity.h>

namespace inference_core {

enum BlobType { kINPUT = 0, kOUTPUT = 1 };

template <typename Type>
inline static std::string VisualVec(const std::vector<Type> &vec)
{
  std::string ret;
  for (const auto &v : vec)
  {
    ret += std::to_string(v) + " ";
  }
  return ret;
}

struct OrtBlobBuffer : public IBlobsBuffer {
public:
  std::pair<void *, DataLocation> GetOuterBlobBuffer(const std::string &blob_name) noexcept override
  {
    if (outer_map_blob2ptr.find(blob_name) == outer_map_blob2ptr.end())
    {
      LOG(ERROR) << "[OrtBlobBuffer] `GetOuterBlobBuffer` Got invalid `blob_name`: " << blob_name;
      return {nullptr, UNKOWN};
    }
    return outer_map_blob2ptr[blob_name];
  }

  bool SetBlobBuffer(const std::string &blob_name,
                     void              *data_ptr,
                     DataLocation       location) noexcept override
  {
    if (inner_map_blob2ptr.find(blob_name) == inner_map_blob2ptr.end())
    {
      LOG(ERROR) << "[OrtBlobBuffer] `SetBlobBuffer` Got invalid `blob_name`: " << blob_name;
      return false;
    }
    outer_map_blob2ptr[blob_name] = {data_ptr, location};
    return true;
  }

  bool SetBlobBuffer(const std::string &blob_name, DataLocation location) noexcept override
  {
    if (inner_map_blob2ptr.find(blob_name) == inner_map_blob2ptr.end())
    {
      LOG(ERROR) << "[OrtBlobBuffer] `SetBlobBuffer` Got invalid `blob_name`: " << blob_name;
      return false;
    }
    outer_map_blob2ptr[blob_name] = {inner_map_blob2ptr[blob_name], location};
    return true;
  }

  bool SetBlobShape(const std::string          &blob_name,
                    const std::vector<int64_t> &shape) noexcept override
  {
    if (map_blob_name2shape.find(blob_name) == map_blob_name2shape.end())
    {
      LOG(ERROR) << "[OrtBlobBuffer] `SetBlobShape` Got invalid `blob_name`: " << blob_name;
      return false;
    }
    const auto &origin_shape = map_blob_name2shape[blob_name];
    if (origin_shape.size() != shape.size())
    {
      const std::string origin_shape_in_str = VisualVec(origin_shape);
      const std::string shape_in_str        = VisualVec(shape);
      LOG(ERROR) << "[OrtBlobBuffer] `SetBlobShape` Got invalid `shape` input. "
                 << "`shape`: " << shape_in_str << "\t"
                 << "`origin_shape`: " << origin_shape_in_str;
      return false;
    }
    map_blob_name2shape[blob_name] = shape;
    return true;
  }

  const std::vector<int64_t> &GetBlobShape(const std::string &blob_name) const noexcept override
  {
    if (map_blob_name2shape.find(blob_name) == map_blob_name2shape.end())
    {
      LOG(ERROR) << "[TrtBlobBuffer] `GetBlobShape` Got invalid `blob_name`: " << blob_name;
      static std::vector<int64_t> empty_shape;
      return empty_shape;
    }
    return map_blob_name2shape.at(blob_name);
  }

  size_t Size() const noexcept override
  {
    return inner_map_blob2ptr.size();
  }

  void Release() noexcept override
  {
    Reset();
    for (float *&ptr : blobs_buffer)
    {
      if (ptr != nullptr)
        delete[] ptr;
      ptr = nullptr;
    }
  }

  void Reset() noexcept override
  {
    map_type2tensors[BlobType::kINPUT].clear();
    map_type2tensors[BlobType::kOUTPUT].clear();
    for (const auto &p_name_ptr : inner_map_blob2ptr)
    {
      outer_map_blob2ptr[p_name_ptr.first] = {p_name_ptr.second, DataLocation::HOST};
    }
  }

  ~OrtBlobBuffer() override
  {
    Release();
  }
  OrtBlobBuffer()                                 = default;
  OrtBlobBuffer(const OrtBlobBuffer &)            = delete;
  OrtBlobBuffer &operator=(const OrtBlobBuffer &) = delete;

  std::unordered_map<std::string, std::pair<void *, DataLocation>> outer_map_blob2ptr;
  std::unordered_map<std::string, void *>                          inner_map_blob2ptr;

  std::vector<float *> blobs_buffer;

  std::unordered_map<BlobType, std::vector<Ort::Value>> map_type2tensors;

  std::unordered_map<std::string, std::vector<int64_t>> map_blob_name2shape;
};

class OrtInferCore : public BaseInferCore {
public:
  ~OrtInferCore() override = default;

  OrtInferCore(const std::string                                            onnx_path,
               const std::unordered_map<std::string, std::vector<int64_t>> &input_blobs_shape,
               const std::unordered_map<std::string, std::vector<int64_t>> &output_blobs_shape,
               const int                                                    num_threads = 0);

  OrtInferCore(const std::string onnx_path, const int num_threads = 0);

  std::shared_ptr<IBlobsBuffer> AllocBlobsBuffer() override;

  InferCoreType GetType()
  {
    return InferCoreType::ONNXRUNTIME;
  }

  std::string GetName()
  {
    return "ort_core";
  }

private:
  bool PreProcess(std::shared_ptr<async_pipeline::IPipelinePackage> buffer) override;

  bool Inference(std::shared_ptr<async_pipeline::IPipelinePackage> buffer) override;

  bool PostProcess(std::shared_ptr<async_pipeline::IPipelinePackage> buffer) override;

private:
  std::unordered_map<std::string, std::vector<int64_t>> ResolveModelInputInformation();

  std::unordered_map<std::string, std::vector<int64_t>> ResolveModelOutputInformation();

  std::unordered_map<std::string, void *> map_blob2ptr_;

  std::shared_ptr<Ort::Env> ort_env_;

  std::shared_ptr<Ort::Session> ort_session_;

  std::unordered_map<std::string, std::vector<int64_t>> map_input_blob_name2shape_;
  std::unordered_map<std::string, std::vector<int64_t>> map_output_blob_name2shape_;
};

OrtInferCore::OrtInferCore(
    const std::string                                            onnx_path,
    const std::unordered_map<std::string, std::vector<int64_t>> &input_blobs_shape,
    const std::unordered_map<std::string, std::vector<int64_t>> &output_blobs_shape,
    const int                                                    num_threads)
{
  // onnxruntime session initialization
  LOG(INFO) << "start initializing onnxruntime session with onnx model {" << onnx_path << "} ...";
  ort_env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, onnx_path.data());
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(num_threads);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(4);
  ort_session_ = std::make_shared<Ort::Session>(*ort_env_, onnx_path.c_str(), session_options);
  LOG(INFO) << "successfully created onnxruntime session!";

  map_input_blob_name2shape_ =
      input_blobs_shape.empty() ? ResolveModelInputInformation() : input_blobs_shape;
  map_output_blob_name2shape_ =
      output_blobs_shape.empty() ? ResolveModelOutputInformation() : output_blobs_shape;

  // show info
  auto func_display_blobs_info =
      [](const std::unordered_map<std::string, std::vector<int64_t>> &blobs_shape) {
        for (const auto &p_name_shape : blobs_shape)
        {
          std::string s_blob_shape;
          for (const auto dim : p_name_shape.second)
          {
            s_blob_shape += std::to_string(dim) + "\t";
          }
          LOG(INFO) << p_name_shape.first << "\tshape: " << s_blob_shape;
        }
      };

  func_display_blobs_info(input_blobs_shape);
  func_display_blobs_info(output_blobs_shape);

  BaseInferCore::Init();
}

std::unordered_map<std::string, std::vector<int64_t>> OrtInferCore::ResolveModelInputInformation()
{
  std::unordered_map<std::string, std::vector<int64_t>> ret;

  OrtAllocator *allocator    = nullptr;
  bool allocator_init_status = Ort::GetApi().GetAllocatorWithDefaultOptions(&allocator) == nullptr;
  CHECK(allocator_init_status);

  const int input_blob_count = ort_session_->GetInputCount();

  for (int i = 0; i < input_blob_count; ++i)
  {
    const auto        blob_info       = ort_session_->GetInputTypeInfo(i);
    const auto        blob_type_shape = blob_info.GetTensorTypeAndShapeInfo();
    const auto        blob_shape      = blob_type_shape.GetShape();
    const auto        blob_name       = ort_session_->GetInputNameAllocated(i, allocator);
    const std::string s_blob_name     = std::string(blob_name.get());

    ret[s_blob_name]              = std::vector<int64_t>();
    std::string s_blob_info       = std::string(blob_name.get()) + ":\t";
    size_t      blob_element_size = 1;
    for (size_t i = 0; i < blob_shape.size(); ++i)
    {
      if (blob_shape[i] < 0)
      {
        throw std::runtime_error(
            "auto resolve onnx model failed! \
                                        for blob shape < 0, please use explicit blob shape constructor!!");
      }
      s_blob_info += "\t" + std::to_string(blob_shape[i]);
      blob_element_size *= blob_shape[i];
      ret[s_blob_name].push_back(blob_shape[i]);
    }
    s_blob_info += "\ttotal elements: " + std::to_string(blob_element_size);
    LOG(INFO) << s_blob_info;
  }

  return ret;
}

std::unordered_map<std::string, std::vector<int64_t>> OrtInferCore::ResolveModelOutputInformation()
{
  std::unordered_map<std::string, std::vector<int64_t>> ret;

  OrtAllocator *allocator    = nullptr;
  bool allocator_init_status = Ort::GetApi().GetAllocatorWithDefaultOptions(&allocator) == nullptr;
  CHECK(allocator_init_status);

  const int output_blob_count = ort_session_->GetOutputCount();

  for (int i = 0; i < output_blob_count; ++i)
  {
    const auto        blob_info       = ort_session_->GetOutputTypeInfo(i);
    const auto        blob_type_shape = blob_info.GetTensorTypeAndShapeInfo();
    const auto        blob_shape      = blob_type_shape.GetShape();
    const auto        blob_name       = ort_session_->GetOutputNameAllocated(i, allocator);
    const std::string s_blob_name     = std::string(blob_name.get());

    ret[s_blob_name]              = std::vector<int64_t>();
    std::string s_blob_info       = std::string(blob_name.get()) + ":\t";
    size_t      blob_element_size = 1;
    for (size_t i = 0; i < blob_shape.size(); ++i)
    {
      if (blob_shape[i] < 0)
      {
        throw std::runtime_error(
            "auto resolve onnx model failed! \
                                        for blob shape < 0, please use explicit blob shape constructor!!");
      }
      s_blob_info += "\t" + std::to_string(blob_shape[i]);
      blob_element_size *= blob_shape[i];
      ret[s_blob_name].push_back(blob_shape[i]);
    }
    s_blob_info += "\ttotal elements: " + std::to_string(blob_element_size);
    LOG(INFO) << s_blob_info;
  }

  return ret;
}

std::shared_ptr<IBlobsBuffer> OrtInferCore::AllocBlobsBuffer()
{
  OrtAllocator *allocator    = nullptr;
  bool allocator_init_status = Ort::GetApi().GetAllocatorWithDefaultOptions(&allocator) == nullptr;
  CHECK(allocator_init_status);

  auto ret = std::make_shared<OrtBlobBuffer>();

  // input blobs
  const int input_blob_count = map_input_blob_name2shape_.size();
  for (int i = 0; i < input_blob_count; ++i)
  {
    const auto        blob_name    = ort_session_->GetInputNameAllocated(i, allocator);
    const std::string s_blob_name  = std::string(blob_name.get());
    const auto       &blob_shape   = map_input_blob_name2shape_[s_blob_name];
    int64_t           element_size = 1;
    for (auto s : blob_shape)
    {
      element_size *= s;
    }

    ret->blobs_buffer.push_back(new float[element_size]);
    ret->inner_map_blob2ptr.insert({s_blob_name, static_cast<void *>(ret->blobs_buffer.back())});
    ret->outer_map_blob2ptr.insert(
        {s_blob_name, {static_cast<void *>(ret->blobs_buffer.back()), DataLocation::HOST}});
    ret->map_blob_name2shape.emplace(s_blob_name, blob_shape);
  }

  // output blobs
  const int output_blob_count = map_output_blob_name2shape_.size();
  for (int i = 0; i < output_blob_count; ++i)
  {
    const auto        blob_name    = ort_session_->GetOutputNameAllocated(i, allocator);
    const std::string s_blob_name  = std::string(blob_name.get());
    const auto       &blob_shape   = map_output_blob_name2shape_[s_blob_name];
    int64_t           element_size = 1;
    for (auto s : blob_shape)
    {
      element_size *= s;
    }

    ret->blobs_buffer.push_back(new float[element_size]);
    ret->inner_map_blob2ptr.insert({s_blob_name, static_cast<void *>(ret->blobs_buffer.back())});
    ret->outer_map_blob2ptr.insert(
        {s_blob_name, {static_cast<void *>(ret->blobs_buffer.back()), DataLocation::HOST}});
    ret->map_blob_name2shape.emplace(s_blob_name, blob_shape);
  }

  return ret;
}

bool OrtInferCore::PreProcess(std::shared_ptr<async_pipeline::IPipelinePackage> buffer)
{
  // 获取内存缓存
  CHECK_STATE(buffer != nullptr, "[ort core] PreProcess got WRONG input data format!");
  auto p_buf = std::dynamic_pointer_cast<OrtBlobBuffer>(buffer->GetInferBuffer());
  CHECK_STATE(p_buf != nullptr, "[ort core] PreProcess got WRONG p_buf data format!");

  OrtBlobBuffer &buf = *p_buf;

  for (const auto &p_name_shape : map_input_blob_name2shape_)
  {
    const auto &s_blob_name        = p_name_shape.first;
    const auto &max_blob_shape     = p_name_shape.second;
    const auto &dynamic_blob_shape = buf.map_blob_name2shape[s_blob_name];

    auto mem_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

    int64_t element_size = sizeof(float);
    for (auto s : dynamic_blob_shape)
    {
      element_size *= s;
    }

    buf.map_type2tensors[BlobType::kINPUT].push_back(
        Ort::Value::CreateTensor(mem_info, buf.outer_map_blob2ptr[s_blob_name].first, element_size,
                                 dynamic_blob_shape.data(), dynamic_blob_shape.size(),
                                 ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
  }

  for (const auto &p_name_shape : map_output_blob_name2shape_)
  {
    const auto &s_blob_name        = p_name_shape.first;
    const auto &max_blob_shape     = p_name_shape.second;
    const auto &dynamic_blob_shape = buf.map_blob_name2shape[s_blob_name];

    auto mem_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

    int64_t element_size = sizeof(float);
    for (auto s : dynamic_blob_shape)
    {
      element_size *= s;
    }

    buf.map_type2tensors[BlobType::kOUTPUT].push_back(
        Ort::Value::CreateTensor(mem_info, buf.outer_map_blob2ptr[s_blob_name].first, element_size,
                                 dynamic_blob_shape.data(), dynamic_blob_shape.size(),
                                 ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
  }
  return true;
}

bool OrtInferCore::Inference(std::shared_ptr<async_pipeline::IPipelinePackage> buffer)
{
  // 获取内存缓存
  CHECK_STATE(buffer != nullptr, "[ort core] Inference got WRONG input data format!");
  auto p_buf = std::dynamic_pointer_cast<OrtBlobBuffer>(buffer->GetInferBuffer());
  CHECK_STATE(p_buf != nullptr, "[ort core] Inference got WRONG p_buf data format!");

  OrtBlobBuffer &buf = *p_buf;

  // 构造推理接口参数
  std::vector<const char *> input_blobs_name;
  std::vector<const char *> output_blobs_name;
  for (const auto &p_name_shape : map_input_blob_name2shape_)
  {
    input_blobs_name.push_back(p_name_shape.first.c_str());
  }
  for (const auto &p_name_shape : map_output_blob_name2shape_)
  {
    output_blobs_name.push_back(p_name_shape.first.c_str());
  }

  // 执行推理
  ort_session_->Run(Ort::RunOptions{nullptr}, input_blobs_name.data(),
                    buf.map_type2tensors[BlobType::kINPUT].data(), input_blobs_name.size(),
                    output_blobs_name.data(), buf.map_type2tensors[BlobType::kOUTPUT].data(),
                    output_blobs_name.size());

  return true;
}

bool OrtInferCore::PostProcess(std::shared_ptr<async_pipeline::IPipelinePackage> buffer)
{
  return true;
}

std::shared_ptr<BaseInferCore> CreateOrtInferCore(
    const std::string                                            onnx_path,
    const std::unordered_map<std::string, std::vector<int64_t>> &input_blobs_shape,
    const std::unordered_map<std::string, std::vector<int64_t>> &output_blobs_shape,
    const int                                                    num_threads)
{
  return std::make_shared<OrtInferCore>(onnx_path, input_blobs_shape, output_blobs_shape,
                                        num_threads);
}

} // namespace inference_core