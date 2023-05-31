#include <opencv2/opencv.hpp>
#include <iostream>
#include "function.h"

using namespace std;
using namespace cv;

int main()
{
    // ��ȡͼƬ
    Mat inputImage = imread("D:/VS2019_Code/Project1/stop_left.jpg", cv::IMREAD_COLOR); // ��ȡRGBͼ��
    if (inputImage.empty())
    {
        printf("could not load image...\n");
        return -1;
    }

    // ת��ΪHSVͼ��
    cv::Mat hsvImage;
    cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);

    // ��ȡ��ɫ����
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
    imshow("��ɫ����", mask);
    waitKey(0);

    // ȥ����ش���
    medianBlur(mask, mask, 5);
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * 2 + 1, 2 * 2 + 1), Point(2, 2));
    Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(2 * 3 + 1, 2 * 3 + 1), Point(3, 3));
    erode(mask, mask, element);//��ʴ    
    dilate(mask, mask, element1);//����
    imshow("�˲�����̬ѧ����", mask);
    waitKey(0);

    // ���
    fillHole(mask, mask);
    imshow("���", mask);
    waitKey(0);

    // ������
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    // ɸѡ�������ҵ���С��Ӿ���
    vector<Rect> boundRect;
    filterContours(contours, boundRect);

    // ��ʾ��С��Ӿ���
    for (const Rect& rect : boundRect) {
        rectangle(inputImage, rect, Scalar(0, 255, 0), 2);
    }

    // ��ʾͼ��
    //imshow("��С��Ӿ���", inputImage);
    //waitKey(0);

    // ����ģ��Ľ�ͨ��־
    Mat templateSrcImg = imread("D:/VS2019_Code/Project1/standard.jpg", cv::IMREAD_GRAYSCALE);

    for (int i = 0; i < boundRect.size(); i++)
    {
        // ��ȡROI
        Mat roi = inputImage(boundRect[i]).clone();

        //1. tmp_roiͼ�� resizeΪ����
        roi.resize(min(roi.rows,roi.cols), min(roi.rows,roi.cols));
        
        // �ҶȻ�����
        Mat grayROI;
        cvtColor(roi, grayROI, cv::COLOR_BGR2GRAY);

        //3. ��ģ��ͼ��ͳһ�ߴ�
        int w = templateSrcImg.cols, h = templateSrcImg.rows;
        resize(grayROI, grayROI, cv::Size(w, h));

        imshow("��������ǰͼ", grayROI);
        waitKey(0);
  

    }
    imshow("ģ��ͼ", templateSrcImg);
    // ��ʾ���ͼ��
    imshow("Result", inputImage);
    waitKey(0);

    return 0;
}
