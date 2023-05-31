#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

//����㷨
void fillHole(const Mat srcBw, Mat& dstBw)
{
	Size m_Size = srcBw.size();
	Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());//��չͼ��
	srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));
	cv::floodFill(Temp, Point(0, 0), Scalar(255));//�������
	Mat cutImg;//�ü���չ��ͼ��
	Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);
	dstBw = srcBw | (~cutImg);
}

//�ж�rect1��rect2�Ƿ��н���  
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


//*����������ɸѡ�����ҵ�������������С��Ӿ��Ρ�
//��������˵��:
//contours��һ���洢�����Ķ�ά������ÿ��������һ������һϵ�е��������
//boundRect��һ���洢��С��Ӿ��ε�������ÿ����С��Ӿ�����һ��Rect�ṹ���ʾ��
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

// ������ƶȶԱ��㷨
double calculateSimilarity(const Mat& image1, const Mat& image2)
{
    Mat grayImage1, grayImage2;
    cvtColor(image1, grayImage1, COLOR_BGR2GRAY);
    cvtColor(image2, grayImage2, COLOR_BGR2GRAY);

    // ʹ��ORB������ȡ��
    Ptr<ORB> orb = ORB::create();

    // ���������ͼ���������
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    orb->detectAndCompute(grayImage1, Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(grayImage2, Mat(), keypoints2, descriptors2);

    // ʹ�ú��������������ƥ��
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // �������ƶȵ÷�
    double maxDist = 0, minDist = numeric_limits<double>::max();
    for (int i = 0; i < matches.size(); i++) {
        double dist = matches[i].distance;
        if (dist < minDist) minDist = dist;
        if (dist > maxDist) maxDist = dist;
    }

    // ѡ��һ�����ʵ���ֵ
    double similarityThreshold = 70; // �������ƶ���ֵΪ70%

    // ������ֵ�������ƶȵ÷�
    double score = 100.0 * (1.0 - minDist / maxDist);

    // �������ƶȵ÷�
    return score;
}