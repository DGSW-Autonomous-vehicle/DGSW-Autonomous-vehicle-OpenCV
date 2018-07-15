//TL.hpp
#ifndef _DEFINE_TL_H_
#define _DEFINE_TL_H_

#define RED 0
#define GREEN 1
#define BLACK 2
#define OTHERS 3

#define LIGHT_NO_CONDITION 0
#define LIGHT_VERTICAL 1
#define LIGHT_HORIZONTAL 2


#include <opencv2/opencv.hpp> 
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

class OpenCV_TL {
private:
	Mat image;
	Mat gray;
	Mat hsv;
	vector<Vec3f> circles;
	vector<Vec3f> TL;
public:
	OpenCV_TL();

	void getLightInfo(int flag);
	int getPixelInfo(Point pt);


};



#endif
