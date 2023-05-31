#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#ifndef FUNCTION_H   //!< ͷ�ļ�����������ֹ���ص���
#define FUNCTION_H 

void fillHole(const Mat srcBw, Mat& dstBw);
bool isInside(Rect rect1, Rect rect2);
void filterContours(vector<vector<Point>>& contours, vector<Rect>& boundRect);
double calculateSSIM(const Mat& img1, const Mat& img2);
double calculateSimilarity(const Mat& image1, const Mat& image2);
#endif
