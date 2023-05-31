#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

//填充算法
void fillHole(const Mat srcBw, Mat& dstBw)
{
	Size m_Size = srcBw.size();
	Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());//延展图像
	srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));
	cv::floodFill(Temp, Point(0, 0), Scalar(255));//填充区域
	Mat cutImg;//裁剪延展的图像
	Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);
	dstBw = srcBw | (~cutImg);
}

//判断rect1与rect2是否有交集  
bool isInside(Rect rect1, Rect rect2)
{
	Rect t = rect1 & rect2;
	if (rect1.area() > rect2.area())
	{
		return false;
	}
	else
	{
		if (t.area() != 0)
			return true;
	}
}


//*对轮廓进行筛选，并找到满足条件的最小外接矩形。
//函数参数说明:
//contours是一个存储轮廓的二维向量，每个轮廓是一个包含一系列点的向量。
//boundRect是一个存储最小外接矩形的向量，每个最小外接矩形用一个Rect结构体表示。
void filterContours(vector<vector<Point>>& contours, vector<Rect>& boundRect)
{
	for (size_t i = 0; i < contours.size(); i++)
	{
		Rect rect = boundingRect(contours[i]);
		if (rect.width > 30 && rect.height > 30)
		{
			boundRect.push_back(rect);
		}
	}
}
double calculateSSIM(const Mat& img1, const Mat& img2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    int d = CV_32F;

    Mat I1, I2;
    img1.convertTo(I1, d);
    img2.convertTo(I2, d);

    Mat I1_2 = I1.mul(I1);
    Mat I2_2 = I2.mul(I2);
    Mat I1_I2 = I1.mul(I2);

    Mat mu1, mu2;
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);

    Mat ssim_map;
    divide(t3, t1.mul(t1) + t2.mul(t2), ssim_map);
    Scalar mssim = mean(ssim_map);

    return (mssim.val[0] + mssim.val[1] + mssim.val[2]) / 3.0;
}

// 添加相似度对比算法
double calculateSimilarity(const Mat& image1, const Mat& image2)
{
    Mat grayImage1, grayImage2;
    cvtColor(image1, grayImage1, COLOR_BGR2GRAY);
    cvtColor(image2, grayImage2, COLOR_BGR2GRAY);

    // 使用ORB特征提取器
    Ptr<ORB> orb = ORB::create();

    // 检测特征点和计算描述符
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    orb->detectAndCompute(grayImage1, Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(grayImage2, Mat(), keypoints2, descriptors2);

    // 使用汉明距离进行特征匹配
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // 计算相似度得分
    double maxDist = 0, minDist = numeric_limits<double>::max();
    for (int i = 0; i < matches.size(); i++) {
        double dist = matches[i].distance;
        if (dist < minDist) minDist = dist;
        if (dist > maxDist) maxDist = dist;
    }

    // 选择一个合适的阈值
    double similarityThreshold = 70; // 设置相似度阈值为70%

    // 根据阈值计算相似度得分
    double score = 100.0 * (1.0 - minDist / maxDist);

    // 返回相似度得分
    return score;
}