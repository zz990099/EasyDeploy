#ifndef __TESTS_ALL_IN_ONE_IMAGE_DRAWER_H
#define __TESTS_ALL_IN_ONE_IMAGE_DRAWER_H

#include "deploy_core/base_detection.h"

#include <opencv2/imgproc.hpp>

#define DRAW_LINE_THICKNESS 2

class ImageDrawHelper {
public:
    ImageDrawHelper() = delete;
    ImageDrawHelper(const std::shared_ptr<cv::Mat> & image) : _image(image) {};

    void drawRect2D(const int center_x,
                    const int center_y,
                    const int width,
                    const int height,
                    const cv::Scalar color = {255, 0, 0})
    {
        cv::rectangle(*_image, 
                    cv::Rect(center_x - width / 2, center_y - height / 2, width, height), 
                    color, 
                    DRAW_LINE_THICKNESS);
    }

    void drawLabel(const int x,
                    const int y,
                    const std::string& label,
                    const cv::Scalar color)
    {
        cv::putText(*_image, label, {x, y}, 0, 2, color);
    }
    
    void drawRect2D(const BBox2D& obj,
                    const cv::Scalar color = {255, 0, 0})
    {
        drawRect2D(obj.x, obj.y, obj.w, obj.h, color);
    }

    void drawRect2DWithLabel(const BBox2D& obj,
                    const std::string& label,
                    const cv::Scalar color = {255, 0, 0})
    {
        drawRect2D(obj.x, obj.y, obj.w, obj.h, color);
        drawLabel(obj.x - obj.w / 2, obj.y - obj.h / 2 - 10, label, color);
    }

    void drawPoint(const std::pair<int,int>& point,
                const cv::Scalar color = {255, 0, 0})
    {
        cv::circle(*_image,
                {point.first, point.second},
                4,
                color,
                -1);
    }


    std::shared_ptr<cv::Mat> getImage() const {
        return _image;
    }
private:
    std::shared_ptr<cv::Mat> _image;
};





#endif