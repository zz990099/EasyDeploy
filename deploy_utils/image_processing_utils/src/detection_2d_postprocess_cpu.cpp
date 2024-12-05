/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-19 18:33:00
 * @LastEditTime: 2024-12-03 15:51:39
 * @FilePath: /EasyDeploy/deploy_utils/image_processing_utils/src/detection_2d_postprocess_cpu.cpp
 */
#include "detection_2d_util/detection_2d_util.h"

namespace detection_2d {

static void DetectionNmsProcess(std::vector<BBox2D> &candidates, std::vector<int> &picked_idxes)
{
  const int candidate_objs_total = candidates.size();

  auto func_rect_inter_area_size = [](const BBox2D &obj_a, const BBox2D &obj_b) {
    cv::Rect_<float> inter = cv::Rect_<float>(obj_a.x, obj_a.y, obj_a.w, obj_a.h) &
                             cv::Rect_<float>(obj_b.x, obj_b.y, obj_b.w, obj_b.h);
    return inter.area();
  };

  std::sort(candidates.begin(), candidates.end(),
            [](const BBox2D &a, const BBox2D &b) { return a.conf > b.conf; });

  for (int i = 0; i < candidate_objs_total; ++i)
  {
    const BBox2D &obj_a = candidates[i];
    bool          keep  = true;
    for (const int picked_idx : picked_idxes)
    {
      const BBox2D &obj_b           = candidates[picked_idx];
      const float   inter_area_size = func_rect_inter_area_size(obj_a, obj_b);
      const float   union_area_size = obj_a.w * obj_a.h + obj_b.w * obj_b.h - inter_area_size;

      if (inter_area_size / union_area_size > 0.6)
      {
        keep = false;
        break;
      }
    }

    if (keep)
    {
      picked_idxes.push_back(i);
    }
  }
}

/**
 * @brief Original Yolov8 output blob shape : [batch, 4 + cls, 8400]
 *
 */
class DetectionPostProcessCPU_Origin : public IDetectionPostProcess {
public:
  DetectionPostProcessCPU_Origin(const int               input_height,
                                 const int               input_width,
                                 const int               cls_number,
                                 const std::vector<int> &downsample_scales);

  void Postprocess(const std::vector<void *> &output_blobs_ptr,
                   std::vector<BBox2D>       &results,
                   float                      conf_threshold,
                   float                      transform_scale) override;

private:
  const int cls_number_;
  int       bboxes_number_;
};

DetectionPostProcessCPU_Origin::DetectionPostProcessCPU_Origin(
    const int               input_height,
    const int               input_width,
    const int               cls_number,
    const std::vector<int> &downsample_scales)
    : cls_number_(cls_number)
{
  bboxes_number_ = 0;
  for (const int s : downsample_scales)
  {
    bboxes_number_ += (input_height / s) * (input_width / s);
  }
}

void DetectionPostProcessCPU_Origin::Postprocess(const std::vector<void *> &_output_blobs_ptr,
                                                 std::vector<BBox2D>       &results,
                                                 float                      conf_threshold,
                                                 float                      transform_scale)
{
  float    *output_blob_ptr  = static_cast<float *>(_output_blobs_ptr[0]);
  const int bbox_info_number = cls_number_ + 4;

  std::vector<BBox2D> candidates;
  for (int i = 0; i < bboxes_number_; ++i)
  {
    BBox2D box;
    box.conf = 0;
    for (int j = 0; j < cls_number_; ++j)
    {
      const size_t offset     = (j + 4) * bboxes_number_ + i;
      const float  local_conf = output_blob_ptr[offset];
      if (local_conf > box.conf)
      {
        box.conf = local_conf;
        box.cls  = j;
      }
    }

    if (box.conf >= conf_threshold)
    {
      box.x = output_blob_ptr[0 * bboxes_number_ + i] / transform_scale;
      box.y = output_blob_ptr[1 * bboxes_number_ + i] / transform_scale;
      box.w = output_blob_ptr[2 * bboxes_number_ + i] / transform_scale;
      box.h = output_blob_ptr[3 * bboxes_number_ + i] / transform_scale;
      candidates.push_back(box);
    }
  }

  std::vector<int> picked_indexes;
  DetectionNmsProcess(candidates, picked_indexes);
  for (const int keep_index : picked_indexes)
  {
    const BBox2D &box = candidates[keep_index];
    results.push_back(box);
  }
}

/**
 * @brief Modified Yolov8 export processing, tranpose the output blob shape to [batch, 8400, 4 +
 * cls]
 *
 */
class DetectionPostProcessCPU_Transpose : public IDetectionPostProcess {
public:
  DetectionPostProcessCPU_Transpose(const int               input_height,
                                    const int               input_width,
                                    const int               cls_number,
                                    const std::vector<int> &downsample_scales);

  void Postprocess(const std::vector<void *> &output_blobs_ptr,
                   std::vector<BBox2D>       &results,
                   float                      conf_threshold,
                   float                      transform_scale) override;

private:
  const int cls_number_;
  int       bboxes_number_;
};

DetectionPostProcessCPU_Transpose::DetectionPostProcessCPU_Transpose(
    const int               input_height,
    const int               input_width,
    const int               cls_number,
    const std::vector<int> &downsample_scales)
    : cls_number_(cls_number)
{
  bboxes_number_ = 0;
  for (const int s : downsample_scales)
  {
    bboxes_number_ += (input_height / s) * (input_width / s);
  }
}

void DetectionPostProcessCPU_Transpose::Postprocess(const std::vector<void *> &_output_blobs_ptr,
                                                    std::vector<BBox2D>       &results,
                                                    float                      conf_threshold,
                                                    float                      transform_scale)
{
  float    *output_blob_ptr  = static_cast<float *>(_output_blobs_ptr[0]);
  const int bbox_info_number = cls_number_ + 4;
  /**
   * @brief In ultralytics v8.2.2, original yolov8 output shape: [batch, 4 + cls_number, 8400]
   * @brief Modify output shape to [batch, 8400, 4 + cls_number]
   */
  std::vector<BBox2D> candidates;
  for (int i = 0; i < bboxes_number_; ++i)
  {
    BBox2D    box;
    const int offset = i * bbox_info_number;

    box.conf = 0;
    for (int j = 0; j < cls_number_; ++j)
    {
      const float local_conf = output_blob_ptr[offset + 4 + j];
      if (local_conf > box.conf)
      {
        box.conf = local_conf;
        box.cls  = j;
      }
    }
    if (box.conf >= conf_threshold)
    {
      box.x = output_blob_ptr[offset] / transform_scale;
      box.y = output_blob_ptr[offset + 1] / transform_scale;
      box.w = output_blob_ptr[offset + 2] / transform_scale;
      box.h = output_blob_ptr[offset + 3] / transform_scale;
      candidates.push_back(box);
    }
  }
  std::vector<int> picked_indexes;
  DetectionNmsProcess(candidates, picked_indexes);
  for (const int keep_index : picked_indexes)
  {
    const BBox2D &box = candidates[keep_index];
    results.push_back(box);
  }
}

/**
 * @brief refer to modified `ultralytics` project by rk team。
 * @brief <rknn example> `https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8`
 * @brief <erase `drl` module>
 * `https://github.com/airockchip/ultralytics_yolov8/blob/main/RKOPT_README.md`
 *
 */
class Yolov8PostProcessCPU_Divide : public IDetectionPostProcess {
public:
  Yolov8PostProcessCPU_Divide(const int               input_height,
                              const int               input_width,
                              const int               cls_number,
                              const std::vector<int> &downsample_scales);

  void Postprocess(const std::vector<void *> &output_blobs_ptr,
                   std::vector<BBox2D>       &results,
                   float                      conf_threshold,
                   float                      transform_scale) override;

private:
  void GenerateCandidates(const std::vector<void *> &output_blobs_ptr,
                          std::vector<BBox2D>       &candidates,
                          float                      conf_threshold,
                          float                      transform_scale);

private:
  const int              cls_number_;
  int                    bboxes_number_;
  const std::vector<int> downsample_scales_;
  const int              input_height_;
  const int              input_width_;
};

Yolov8PostProcessCPU_Divide::Yolov8PostProcessCPU_Divide(const int               input_height,
                                                         const int               input_width,
                                                         const int               cls_number,
                                                         const std::vector<int> &downsample_scales)
    : cls_number_(cls_number),
      downsample_scales_(downsample_scales),
      input_height_(input_height),
      input_width_(input_width)
{
  bboxes_number_ = 0;
  for (const int s : downsample_scales)
  {
    bboxes_number_ += (input_height / s) * (input_width / s);
  }
}

void Yolov8PostProcessCPU_Divide::GenerateCandidates(const std::vector<void *> &output_blobs_ptr,
                                                     std::vector<BBox2D>       &candidates,
                                                     float                      conf_threshold,
                                                     float                      transform_scale)
{
  const int level_count = downsample_scales_.size();
  for (int l = 0; l < level_count; ++l)
  {
    const int    l_h                = input_height_ / downsample_scales_[l];
    const int    l_w                = input_width_ / downsample_scales_[l];
    const int    total_element_size = l_h * l_w;
    const float *box_ptr            = static_cast<float *>(output_blobs_ptr[l * level_count]);
    const float *cls_ptr            = static_cast<float *>(output_blobs_ptr[l * level_count + 1]);
    const float *cls_reduce_ptr     = static_cast<float *>(output_blobs_ptr[l * level_count + 2]);

    for (int r = 0; r < l_h; ++r)
    {
      for (int c = 0; c < l_w; ++c)
      {
        const size_t grid_offset = r * l_w + c;
        // use cls_reduce
        if (cls_reduce_ptr[grid_offset] < conf_threshold)
          continue;

        // find max confidence
        float max_confidence = 0;
        int   max_cls_index  = 0;
        for (int i = 0; i < cls_number_; ++i)
        {
          float conf = cls_ptr[grid_offset + total_element_size * i];
          if (conf > conf_threshold && conf > max_confidence)
          {
            max_confidence = conf;
            max_cls_index  = i;
          }
        }

        // calculate bbox
        if (max_confidence > conf_threshold)
        {
          static const int   dfl_len            = 16; // TODO
          static const int   bbox_params_number = 64; // TODO
          std::vector<float> before_dfl(bbox_params_number);
          for (int i = 0; i < bbox_params_number; ++i)
          {
            before_dfl[i] = box_ptr[grid_offset + total_element_size * i];
          }
          // compute dfl
          std::vector<float> box(4);
          for (int i = 0; i < 4; ++i)
          {
            std::vector<float> exp_t(dfl_len);
            float              exp_sum = 0;
            float              acc_sum = 0;
            for (int j = 0; j < dfl_len; ++j)
            {
              exp_t[j] = exp(before_dfl[j + i * dfl_len]);
              exp_sum += exp_t[j];
            }
            for (int j = 0; j < dfl_len; ++j)
            {
              acc_sum += exp_t[j] / exp_sum * j;
            }
            box[i] = acc_sum;
          }
          float x1, y1, x2, y2, w, h;
          x1 = (-box[0] + c + 0.5) * downsample_scales_[l] / transform_scale;
          y1 = (-box[1] + r + 0.5) * downsample_scales_[l] / transform_scale;
          x2 = (box[2] + c + 0.5) * downsample_scales_[l] / transform_scale;
          y2 = (box[3] + r + 0.5) * downsample_scales_[l] / transform_scale;
          w  = x2 - x1;
          h  = y2 - y1;
          BBox2D candidate;
          candidate.x    = (x1 + x2) / 2;
          candidate.y    = (y1 + y2) / 2;
          candidate.w    = w;
          candidate.h    = h;
          candidate.conf = max_confidence;
          candidate.cls  = max_cls_index;
          candidates.push_back(candidate);
        }
      }
    }
  }
}

void Yolov8PostProcessCPU_Divide::Postprocess(const std::vector<void *> &output_blobs_ptr,
                                              std::vector<BBox2D>       &results,
                                              float                      conf_threshold,
                                              float                      transform_scale)
{
  // box, cls, cls_reduce
  CHECK(output_blobs_ptr.size() == downsample_scales_.size() * 3);
  // generate candidates
  std::vector<BBox2D> candidates;
  GenerateCandidates(output_blobs_ptr, candidates, conf_threshold, transform_scale);

  std::vector<int> picked_indexes;
  DetectionNmsProcess(candidates, picked_indexes);
  for (const int keep_index : picked_indexes)
  {
    const BBox2D &box = candidates[keep_index];
    results.push_back(box);
  }
}

std::shared_ptr<IDetectionPostProcess> CreateYolov8PostProcessCpuOrigin(
    const int               input_height,
    const int               input_width,
    const int               cls_number,
    const std::vector<int> &downsample_scales)
{
  return std::make_shared<DetectionPostProcessCPU_Origin>(input_height, input_width, cls_number,
                                                          downsample_scales);
}

std::shared_ptr<IDetectionPostProcess> CreateYolov8PostProcessCpuTranspose(
    const int               input_height,
    const int               input_width,
    const int               cls_number,
    const std::vector<int> &downsample_scales)
{
  return std::make_shared<DetectionPostProcessCPU_Transpose>(input_height, input_width, cls_number,
                                                             downsample_scales);
}

std::shared_ptr<IDetectionPostProcess> CreateYolov8PostProcessCpuDivide(
    const int               input_height,
    const int               input_width,
    const int               cls_number,
    const std::vector<int> &downsample_scales)
{
  return std::make_shared<Yolov8PostProcessCPU_Divide>(input_height, input_width, cls_number,
                                                       downsample_scales);
}

struct Detection2DYolov8PostprocessParams {
  int              input_height;
  int              input_width;
  int              cls_number;
  std::vector<int> downsample_scales;
};

class Detection2DYolov8PostprocessOriginFactory : public BaseDetectionPostprocessFactory {
public:
  Detection2DYolov8PostprocessOriginFactory(const Detection2DYolov8PostprocessParams &params)
      : params_(params)
  {}

  std::shared_ptr<IDetectionPostProcess> Create() override
  {
    return CreateYolov8PostProcessCpuOrigin(params_.input_height, params_.input_width,
                                            params_.cls_number, params_.downsample_scales);
  }

private:
  const Detection2DYolov8PostprocessParams params_;
};

std::shared_ptr<BaseDetectionPostprocessFactory> CreateYolov8PostProcessCpuOriginFactory(
    const int               input_height,
    const int               input_width,
    const int               cls_number,
    const std::vector<int> &downsample_scales)
{
  Detection2DYolov8PostprocessParams params;
  params.input_height      = input_height;
  params.input_width       = input_width;
  params.cls_number        = cls_number;
  params.downsample_scales = downsample_scales;

  return std::make_shared<Detection2DYolov8PostprocessOriginFactory>(params);
}

class Detection2DYolov8PostprocessTransposeFactory : public BaseDetectionPostprocessFactory {
public:
  Detection2DYolov8PostprocessTransposeFactory(const Detection2DYolov8PostprocessParams &params)
      : params_(params)
  {}

  std::shared_ptr<IDetectionPostProcess> Create() override
  {
    return CreateYolov8PostProcessCpuTranspose(params_.input_height, params_.input_width,
                                               params_.cls_number, params_.downsample_scales);
  }

private:
  const Detection2DYolov8PostprocessParams params_;
};

std::shared_ptr<BaseDetectionPostprocessFactory> CreateYolov8PostProcessCpuTransposeFactory(
    const int               input_height,
    const int               input_width,
    const int               cls_number,
    const std::vector<int> &downsample_scales)
{
  Detection2DYolov8PostprocessParams params;
  params.input_height      = input_height;
  params.input_width       = input_width;
  params.cls_number        = cls_number;
  params.downsample_scales = downsample_scales;

  return std::make_shared<Detection2DYolov8PostprocessTransposeFactory>(params);
}

class Detection2DYolov8PostprocessDivideFactory : public BaseDetectionPostprocessFactory {
public:
  Detection2DYolov8PostprocessDivideFactory(const Detection2DYolov8PostprocessParams &params)
      : params_(params)
  {}

  std::shared_ptr<IDetectionPostProcess> Create() override
  {
    return CreateYolov8PostProcessCpuDivide(params_.input_height, params_.input_width,
                                            params_.cls_number, params_.downsample_scales);
  }

private:
  const Detection2DYolov8PostprocessParams params_;
};

std::shared_ptr<BaseDetectionPostprocessFactory> CreateYolov8PostProcessCpuDivideFactory(
    const int               input_height,
    const int               input_width,
    const int               cls_number,
    const std::vector<int> &downsample_scales)
{
  Detection2DYolov8PostprocessParams params;
  params.input_height      = input_height;
  params.input_width       = input_width;
  params.cls_number        = cls_number;
  params.downsample_scales = downsample_scales;

  return std::make_shared<Detection2DYolov8PostprocessDivideFactory>(params);
}

} // namespace detection_2d