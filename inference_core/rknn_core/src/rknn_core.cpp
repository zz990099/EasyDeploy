#include "rknn_core/rknn_core.h"

#include <unordered_map>

#include <rknn_api.h>

namespace inference_core {

static std::unordered_map<RknnInputTensorType, rknn_tensor_type> map_type_my2rk{
    {RknnInputTensorType::RK_UINT8, RKNN_TENSOR_UINT8},
    {RknnInputTensorType::RK_INT8, RKNN_TENSOR_INT8},
    {RknnInputTensorType::RK_FLOAT16, RKNN_TENSOR_FLOAT16},
    {RknnInputTensorType::RK_FLOAT32, RKNN_TENSOR_FLOAT32},
    {RknnInputTensorType::RK_UINT32, RKNN_TENSOR_UINT32},
    {RknnInputTensorType::RK_INT32, RKNN_TENSOR_INT32},
    {RknnInputTensorType::RK_INT64, RKNN_TENSOR_INT64},
};

static std::unordered_map<rknn_tensor_type, int> map_rknn_type2size_{
    {RKNN_TENSOR_INT8, 1},    {RKNN_TENSOR_UINT8, 1}, {RKNN_TENSOR_FLOAT16, 4},
    {RKNN_TENSOR_FLOAT32, 4}, {RKNN_TENSOR_INT32, 4}, {RKNN_TENSOR_UINT32, 4},
    {RKNN_TENSOR_INT64, 8}};

static std::unordered_map<rknn_tensor_type, rknn_tensor_type> map_rknn_type2type{
    {RKNN_TENSOR_INT8, RKNN_TENSOR_UINT8},      {RKNN_TENSOR_UINT8, RKNN_TENSOR_UINT8},
    {RKNN_TENSOR_FLOAT16, RKNN_TENSOR_FLOAT32}, {RKNN_TENSOR_FLOAT32, RKNN_TENSOR_FLOAT32},
    {RKNN_TENSOR_INT32, RKNN_TENSOR_INT32},     {RKNN_TENSOR_UINT32, RKNN_TENSOR_UINT32},
    {RKNN_TENSOR_INT64, RKNN_TENSOR_INT64}};

class RknnBlobBuffer : public IBlobsBuffer {
public:
  std::pair<void *, DataLocation> GetOuterBlobBuffer(const std::string &blob_name) noexcept override
  {
    if (outer_map_blob2ptr.find(blob_name) == outer_map_blob2ptr.end())
    {
      LOG(ERROR) << "[RknnBlobBuffer] `GetOuterBlobBuffer` Got invalid `blob_name`: " << blob_name;
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
      LOG(ERROR) << "[RknnBlobBuffer] `SetBlobBuffer` Got invalid `blob_name`: " << blob_name;
      return false;
    }
    outer_map_blob2ptr[blob_name] = {data_ptr, location};
    return true;
  }

  bool SetBlobBuffer(const std::string &blob_name, DataLocation location) noexcept override
  {
    if (inner_map_blob2ptr.find(blob_name) == inner_map_blob2ptr.end())
    {
      LOG(ERROR) << "[RknnBlobBuffer] `SetBlobBuffer` Got invalid `blob_name`: " << blob_name;
      return false;
    }
    outer_map_blob2ptr[blob_name] = {inner_map_blob2ptr[blob_name], location};
    return true;
  }

  bool SetBlobShape(const std::string          &blob_name,
                    const std::vector<int64_t> &shape) noexcept override
  {
    LOG(WARNING) << "[RknnBlobBuffer] `SetBlobShape` dynamic input shape not supported!!!";
    return false;
  }

  const std::vector<int64_t> &GetBlobShape(const std::string &blob_name) const noexcept override
  {
    if (map_blob_name2shape.find(blob_name) == map_blob_name2shape.end())
    {
      LOG(ERROR) << "[RknnBlobBuffer] `GetBlobShape` Got invalid `blob_name`: " << blob_name;
      static std::vector<int64_t> empty_shape;
      return empty_shape;
    }
    return map_blob_name2shape.at(blob_name);
  }

  size_t Size() const noexcept override
  {
    return outer_map_blob2ptr.size();
  }

  void Release() noexcept override
  {
    for (const auto &p_name_ptr : input_blobs_ptr)
    {
      if (p_name_ptr.second != nullptr)
      {
        delete[] p_name_ptr.second;
      }
    }
    for (const auto &p_name_ptr : output_blobs_ptr)
    {
      if (p_name_ptr.second != nullptr)
      {
        delete[] p_name_ptr.second;
      }
    }
    outer_map_blob2ptr.clear();
    inner_map_blob2ptr.clear();
    input_blobs_ptr.clear();
    output_blobs_ptr.clear();
  }

  void Reset() noexcept override
  {
    for (const auto &p_name_ptr : inner_map_blob2ptr)
    {
      outer_map_blob2ptr[p_name_ptr.first] = {p_name_ptr.second, DataLocation::HOST};
    }
  }

  ~RknnBlobBuffer() override
  {
    Release();
  }
  //
  RknnBlobBuffer()                                  = default;
  RknnBlobBuffer(const RknnBlobBuffer &)            = delete;
  RknnBlobBuffer &operator=(const RknnBlobBuffer &) = delete;

  //
  std::unordered_map<std::string, std::pair<void *, DataLocation>> outer_map_blob2ptr;
  std::unordered_map<std::string, void *>                          inner_map_blob2ptr;

  //
  std::unordered_map<std::string, u_char *> input_blobs_ptr;
  std::unordered_map<std::string, float *>  output_blobs_ptr;

  //
  std::vector<rknn_input>  device_buffer_input;
  std::vector<rknn_output> device_buffer_output;

  //
  std::unordered_map<std::string, std::vector<int64_t>> map_blob_name2shape;

  //
  std::future<bool> async_infer_handle_;
};

class RknnInferCore : public BaseInferCore {
public:
  RknnInferCore(std::string                                                 model_path,
                const std::unordered_map<std::string, RknnInputTensorType> &map_blob_type,
                const int                                                   mem_buf_size     = 5,
                const int                                                   parallel_ctx_num = 1);

  ~RknnInferCore() override;

  InferCoreType GetType()
  {
    return InferCoreType::RKNN;
  }

  std::string GetName()
  {
    return "rknn_core";
  }

private:
  bool PreProcess(std::shared_ptr<async_pipeline::IPipelinePackage> buffer) override;

  bool Inference(std::shared_ptr<async_pipeline::IPipelinePackage> buffer) override;

  bool PostProcess(std::shared_ptr<async_pipeline::IPipelinePackage> buffer) override;

private:
  std::shared_ptr<IBlobsBuffer> AllocBlobsBuffer() override;

  std::shared_ptr<RknnBlobBuffer> _AllocBlobsBuffer();

  size_t ReadModelFromFile(const std::string &model_path, void **model_data);

  void ResolveModelInformation(
      const std::unordered_map<std::string, RknnInputTensorType> &map_blob_type);

private:
  //
  std::vector<rknn_context> rknn_ctx_parallel_;
  //
  BlockQueue<int> bq_ctx_;

  //
  int blob_input_number_;
  int blob_output_number_;

  std::vector<rknn_tensor_attr> blob_attr_input_;
  std::vector<rknn_tensor_attr> blob_attr_output_;

  std::unordered_map<std::string, std::vector<int64_t>> map_input_blob_name2shape_;
  std::unordered_map<std::string, std::vector<int64_t>> map_output_blob_name2shape_;
  std::vector<int64_t>                                  blob_element_size_input_;
  std::vector<int64_t>                                  blob_element_size_output_;
  std::vector<rknn_tensor_type>                         blob_tensor_type_input_;
};

RknnInferCore::RknnInferCore(
    std::string                                                 model_path,
    const std::unordered_map<std::string, RknnInputTensorType> &map_blob_type,
    const int                                                   mem_buf_size,
    const int                                                   parallel_ctx_num)
    : bq_ctx_(parallel_ctx_num)
{
  if (parallel_ctx_num <= 0)
  {
    throw std::invalid_argument("[rknn core] Got Invalid ctx_num: " +
                                std::to_string(parallel_ctx_num));
  }

  void  *model_data           = nullptr;
  size_t model_data_byte_size = ReadModelFromFile(model_path, &model_data);
  if (model_data == nullptr)
  {
    throw std::runtime_error("[rknn_core] Failed to read model from file: " + model_path);
  }
  LOG(INFO) << "[rknn core] initilize using " << parallel_ctx_num << " ctx instances";
  rknn_ctx_parallel_.resize(parallel_ctx_num);
  for (int i = 0; i < parallel_ctx_num; ++i)
  {
    if (rknn_init(&rknn_ctx_parallel_[i], model_data, model_data_byte_size, 0, NULL) != RKNN_SUCC)
    {
      throw std::runtime_error("[rknn_core] Failed to init rknn_ctx [ " + std::to_string(i) + " ]");
    }
    bq_ctx_.BlockPush(i);
  }
  
  rknn_sdk_version version;
  auto             ret =
      rknn_query(rknn_ctx_parallel_[0], RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
  if (ret < 0)
  {
    LOG(ERROR) << "[rknn core] Failed to get rknn sdk version info!!!";
  } else
  {
    LOG(INFO) << "sdk version: " << version.api_version
              << ", driver version: " << version.drv_version;
  }

  free(model_data);

  ResolveModelInformation(map_blob_type);

  BaseInferCore::Init(mem_buf_size);
}

size_t RknnInferCore::ReadModelFromFile(const std::string &model_path, void **model_data)
{
  FILE *fp = fopen(model_path.c_str(), "rb");
  if (fp == NULL)
  {
    printf("fopen %s fail!\n", model_path.c_str());
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  size_t file_size = ftell(fp);
  char  *data      = (char *)malloc(file_size + 1);
  data[file_size]  = 0;
  fseek(fp, 0, SEEK_SET);
  if (file_size != fread(data, 1, file_size, fp))
  {
    printf("fread %s fail!\n", model_path.c_str());
    free(data);
    fclose(fp);
    return -1;
  }
  if (fp)
  {
    fclose(fp);
  }
  *model_data = data;
  return file_size;
}

RknnInferCore::~RknnInferCore()
{
  //////////////////////////// IMPORTANT /////////////////////////////////
  for (size_t i = 0; i < rknn_ctx_parallel_.size(); ++i)
  {
    bq_ctx_.Take();
  }

  for (auto &rknn_context : rknn_ctx_parallel_)
  {
    if (rknn_destroy(rknn_context) != RKNN_SUCC)
    {
      LOG(ERROR) << "[rknn core] In deconstructor destroy rknn ctx failed!!!";
    }
  }
  rknn_ctx_parallel_.clear();
}

std::shared_ptr<IBlobsBuffer> RknnInferCore::AllocBlobsBuffer()
{
  return _AllocBlobsBuffer();
}

void RknnInferCore::ResolveModelInformation(
    const std::unordered_map<std::string, RknnInputTensorType> &map_blob_type)
{
  rknn_input_output_num rknn_io_num;
  if (rknn_query(rknn_ctx_parallel_[0], RKNN_QUERY_IN_OUT_NUM, &rknn_io_num, sizeof(rknn_io_num)) !=
      RKNN_SUCC)
  {
    throw std::runtime_error("[rknn core] Failed to execute in_out_num `rknn_query`");
  }
  LOG(INFO) << "model input blob num: " << rknn_io_num.n_input
            << "\toutput blob num: " << rknn_io_num.n_output;

  blob_input_number_  = rknn_io_num.n_input;
  blob_output_number_ = rknn_io_num.n_output;
  blob_element_size_input_.resize(blob_input_number_);
  blob_element_size_output_.resize(blob_output_number_);
  blob_attr_input_.resize(blob_input_number_);
  blob_attr_output_.resize(blob_output_number_);

  // input blob
  blob_tensor_type_input_.resize(blob_input_number_);
  for (int i = 0; i < blob_input_number_; ++i)
  {
    blob_attr_input_[i].index = i;
    if (rknn_query(rknn_ctx_parallel_[0], RKNN_QUERY_INPUT_ATTR, &(blob_attr_input_[i]),
                   sizeof(rknn_tensor_attr)) != RKNN_SUCC)
    {
      throw std::runtime_error("[rknn core] Failed to execute input `rknn_query`");
    }
    const std::string s_blob_name = blob_attr_input_[i].name;
    //
    rknn_tensor_type blob_type;
    if (map_blob_type.find(s_blob_name) != map_blob_type.end())
    {
      blob_type = map_type_my2rk[map_blob_type.at(s_blob_name)];
    } else
    {
      blob_type = blob_attr_input_[i].type;
    }

    if (map_rknn_type2size_.find(blob_type) == map_rknn_type2size_.end())
    {
      LOG(ERROR) << "[rknn core] blob_name: " << s_blob_name << ", blob_type : " << blob_type
                 << " NOT FOUND in `map_rknn_type2size_`";
      throw std::runtime_error("[rknn core] Failed to resolve model information!!!");
    }
    blob_tensor_type_input_[i]    = map_rknn_type2type[blob_type];
    const int blob_type_byte_size = map_rknn_type2size_[blob_type];

    std::vector<int64_t> blob_shape;
    size_t               blob_element_size = blob_type_byte_size;
    std::string          s_blob_info       = s_blob_name;
    for (size_t j = 0; j < blob_attr_input_[i].n_dims; ++j)
    {
      s_blob_info += "\t" + std::to_string(blob_attr_input_[i].dims[j]);
      blob_element_size *= blob_attr_input_[i].dims[j];
      blob_shape.push_back(blob_attr_input_[i].dims[j]);
    }
    LOG(INFO) << s_blob_info;
    LOG(INFO) << "blob fmt: " << get_format_string(blob_attr_input_[i].fmt)
              << ",  type: " << get_type_string(blob_tensor_type_input_[i]);
    map_input_blob_name2shape_[s_blob_name] = blob_shape;
    blob_element_size_input_[i]             = blob_element_size;
  }

  // output blob
  for (int i = 0; i < blob_output_number_; ++i)
  {
    blob_attr_output_[i].index = i;
    if (rknn_query(rknn_ctx_parallel_[0], RKNN_QUERY_OUTPUT_ATTR, &(blob_attr_output_[i]),
                   sizeof(rknn_tensor_attr)) != RKNN_SUCC)
    {
      throw std::runtime_error("[rknn core] Failed to execute output `rknn_query`");
    }
    const std::string    s_blob_name = blob_attr_output_[i].name;
    std::vector<int64_t> blob_shape;
    size_t               blob_element_size = 1;
    std::string          s_blob_info       = blob_attr_output_[i].name;
    for (size_t j = 0; j < blob_attr_output_[i].n_dims; ++j)
    {
      s_blob_info += "\t" + std::to_string(blob_attr_output_[i].dims[j]);
      blob_element_size *= blob_attr_output_[i].dims[j];
      blob_shape.push_back(blob_attr_output_[i].dims[j]);
    }
    LOG(INFO) << s_blob_info;
    LOG(INFO) << "blob fmt: " << blob_attr_output_[i].fmt
              << ",  type: " << blob_attr_output_[i].type;

    map_output_blob_name2shape_[s_blob_name] = blob_shape;
    blob_element_size_output_[i]             = blob_element_size;
  }
}

std::shared_ptr<RknnBlobBuffer> RknnInferCore::_AllocBlobsBuffer()
{
  auto ret = std::make_shared<RknnBlobBuffer>();

  ret->device_buffer_input.resize(blob_input_number_);
  for (int i = 0; i < blob_input_number_; ++i)
  {
    const std::string s_blob_name  = blob_attr_input_[i].name;
    int64_t           element_size = blob_element_size_input_[i];

    u_char *buf = new u_char[element_size];
    ret->input_blobs_ptr.insert({s_blob_name, buf});
    ret->outer_map_blob2ptr.insert({s_blob_name, {buf, DataLocation::HOST}});
    ret->inner_map_blob2ptr.insert({s_blob_name, buf});

    ret->map_blob_name2shape.insert({s_blob_name, map_input_blob_name2shape_[s_blob_name]});

    //
    ret->device_buffer_input[i].index = i;
    ret->device_buffer_input[i].fmt   = blob_attr_input_[i].fmt;
    ret->device_buffer_input[i].type  = blob_tensor_type_input_[i];
    ret->device_buffer_input[i].size  = element_size;
  }

  ret->device_buffer_output.resize(blob_output_number_);
  for (int i = 0; i < blob_output_number_; ++i)
  {
    const std::string s_blob_name  = blob_attr_output_[i].name;
    int64_t           element_size = blob_element_size_output_[i];

    float *out_buf = new float[element_size];
    ret->output_blobs_ptr.insert({s_blob_name, out_buf});

    ret->outer_map_blob2ptr.insert({s_blob_name, {out_buf, DataLocation::HOST}});
    ret->inner_map_blob2ptr.insert({s_blob_name, out_buf});

    ret->map_blob_name2shape.insert({s_blob_name, map_output_blob_name2shape_[s_blob_name]});

    //
    ret->device_buffer_output[i].index       = i;
    ret->device_buffer_output[i].is_prealloc = true;
    ret->device_buffer_output[i].want_float  = true;
    ret->device_buffer_output[i].size        = element_size * sizeof(float);
  }

  return ret;
}

bool RknnInferCore::PreProcess(std::shared_ptr<async_pipeline::IPipelinePackage> buffer)
{
  //
  auto p_buf = std::dynamic_pointer_cast<RknnBlobBuffer>(buffer->GetInferBuffer());
  CHECK_STATE(p_buf != nullptr, "[rknn core] PreProcess got wrong input data format!");

  RknnBlobBuffer &buf = *p_buf;

  for (int i = 0; i < blob_input_number_; ++i)
  {
    const std::string s_blob_name = blob_attr_input_[i].name;

    void *outer_ptr                = buf.outer_map_blob2ptr[s_blob_name].first;
    buf.device_buffer_input[i].buf = outer_ptr;
  }

  for (int i = 0; i < blob_output_number_; ++i)
  {
    const std::string s_blob_name = blob_attr_output_[i].name;

    void *ptr                       = buf.outer_map_blob2ptr[s_blob_name].first;
    buf.device_buffer_output[i].buf = ptr;
  }

  return true;
}

#define RKNN_CHECK_STATE(state, hint) \
  {                                   \
    if (!(state))                     \
    {                                 \
      LOG(ERROR) << (hint);           \
      bq_ctx_.BlockPush(index);       \
      return false;                   \
    }                                 \
  }

bool RknnInferCore::Inference(std::shared_ptr<async_pipeline::IPipelinePackage> buffer)
{
  //
  auto p_buf = std::dynamic_pointer_cast<RknnBlobBuffer>(buffer->GetInferBuffer());
  CHECK_STATE(p_buf != nullptr, "[rknn core] Inference got wrong input data format!");

  auto func_async_infer = [this, p_buf](int index) -> bool {
    //
    RKNN_CHECK_STATE(rknn_inputs_set(rknn_ctx_parallel_[index], blob_input_number_,
                                     p_buf->device_buffer_input.data()) == RKNN_SUCC,
                     "[rknn core] Inference `rknn_inputs_set` execute failed!!!");
    RKNN_CHECK_STATE(rknn_run(rknn_ctx_parallel_[index], nullptr) == RKNN_SUCC,
                     "[rknn core] Inference `rknn_run` execute failed!!!");
    RKNN_CHECK_STATE(rknn_outputs_get(rknn_ctx_parallel_[index], blob_output_number_,
                                      p_buf->device_buffer_output.data(), nullptr) == RKNN_SUCC,
                     "[rknn core] Inference `rknn_outputs_get` execute failed!!!");

    RKNN_CHECK_STATE(rknn_outputs_release(rknn_ctx_parallel_[index], blob_output_number_,
                                          p_buf->device_buffer_output.data()) == RKNN_SUCC,
                     "[rknn core] Inference `rknn_outputs_release failed!!!");

    bq_ctx_.BlockPush(index);
    return true;
  };
  auto ctx = bq_ctx_.Take();
  if (!ctx.has_value())
  {
    return false;
  }
  p_buf->async_infer_handle_ = std::async(func_async_infer, ctx.value());

  return true;
}

bool RknnInferCore::PostProcess(std::shared_ptr<async_pipeline::IPipelinePackage> buffer)
{
  auto p_buf = std::dynamic_pointer_cast<RknnBlobBuffer>(buffer->GetInferBuffer());
  CHECK_STATE(p_buf != nullptr, "[rknn core] PostProcess got wrong input data format!");

  CHECK_STATE(p_buf->async_infer_handle_.get(),
              "[rknn core] async infer handle got `false` from async process!");

  return true;
}

std::shared_ptr<BaseInferCore> CreateRknnInferCore(
    std::string                                                 model_path,
    const std::unordered_map<std::string, RknnInputTensorType> &map_blob_type,
    const int                                                   mem_buf_size,
    const int                                                   parallel_ctx_num)
{
  return std::make_shared<RknnInferCore>(model_path, map_blob_type, mem_buf_size, parallel_ctx_num);
}

} // namespace inference_core