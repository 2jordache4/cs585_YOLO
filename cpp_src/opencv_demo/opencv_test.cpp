#include <iostream>
#include <opencv2/opencv.hpp>

int main(void){
	cv::Mat image = cv::imread("./image.png");
	
	if (image.empty()){
		std::cout << "failed to open" << std::endl;
		return 0;
	}

	cv::namedWindow("Display wind", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display wind", image);
	cv::waitKey(0);
	
	return 0;
}