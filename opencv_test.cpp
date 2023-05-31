#include <opencv2/opencv.hpp>
#include <iostream>
#include "function.h"

using namespace std;
using namespace cv;

int main()
{
    // 读取图片
    Mat inputImage = imread("D:/VS2019_Code/Project1/stop_left.jpg", cv::IMREAD_COLOR); // 读取RGB图像
    if (inputImage.empty())
    {
        printf("could not load image...\n");
        return -1;
    }

    // 转换为HSV图像
    cv::Mat hsvImage;
    cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);

    // 提取红色部分
    Mat mask = Mat::zeros(inputImage.size(), CV_8UC1);
    for (int y = 0; y < hsvImage.rows; y++)
    {
        for (int x = 0; x < hsvImage.cols; x++)
        {
            Vec3b hsv = hsvImage.at<Vec3b>(y, x);
            if ((hsv[0] >= 168 && hsv[0] <= 180) || (hsv[0] >= 0 && hsv[0] <= 10) &&
                (hsv[1] >= 70 && hsv[1] <= 255) && (hsv[2] >= 70 && hsv[2] <= 255))
            {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }
    imshow("红色部分", mask);
    waitKey(0);

    // 去噪相关处理
    medianBlur(mask, mask, 5);
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * 2 + 1, 2 * 2 + 1), Point(2, 2));
    Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(2 * 3 + 1, 2 * 3 + 1), Point(3, 3));
    erode(mask, mask, element);//腐蚀    
    dilate(mask, mask, element1);//膨胀
    imshow("滤波与形态学处理", mask);
    waitKey(0);

    // 填充
    fillHole(mask, mask);
    imshow("填充", mask);
    waitKey(0);

    // 找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    // 筛选轮廓并找到最小外接矩形
    vector<Rect> boundRect;
    filterContours(contours, boundRect);

    // 显示最小外接矩形
    for (const Rect& rect : boundRect) {
        rectangle(inputImage, rect, Scalar(0, 255, 0), 2);
    }

    // 显示图像
    //imshow("最小外接矩形", inputImage);
    //waitKey(0);

    // 加载模板的交通标志
    Mat templateSrcImg = imread("D:/VS2019_Code/Project1/standard.jpg", cv::IMREAD_GRAYSCALE);

    for (int i = 0; i < boundRect.size(); i++)
    {
        // 提取ROI
        Mat roi = inputImage(boundRect[i]).clone();

        //1. tmp_roi图像 resize为方形
        roi.resize(min(roi.rows,roi.cols), min(roi.rows,roi.cols));
        
        // 灰度化处理
        Mat grayROI;
        cvtColor(roi, grayROI, cv::COLOR_BGR2GRAY);

        //3. 与模板图像统一尺寸
        int w = templateSrcImg.cols, h = templateSrcImg.rows;
        resize(grayROI, grayROI, cv::Size(w, h));

        imshow("处理后的提前图", grayROI);
        waitKey(0);
  

    }
    imshow("模板图", templateSrcImg);
    // 显示结果图像
    imshow("Result", inputImage);
    waitKey(0);

    return 0;
}
