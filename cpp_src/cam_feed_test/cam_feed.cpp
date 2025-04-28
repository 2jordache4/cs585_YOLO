// cam test program
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include <iostream>
#include <chrono>

std::string form_pipeline(int width, int height, int framerate, int dis_width, int dis_height){
	/*
	return
		" libcamerasrc ! video/x-raw, "
		" width=(int)" + std::to_string(width) + ", "
		" height=(int)" + std::to_string(height) + ", "
		" framerate=(fraction)" + std::to_string(framerate) + ", "
		" videoconvert ! videoscale !"
		" video/x-raw,"
		" width=(int)" + std::to_string(dis_width) + ", "
		" height=(int)" + std::to_string(dis_height) + ", "
		" ! queue leaky=downstream max-size-buffers=1 ! appsink sync=false";
	*/
	return
		" libcamerasrc ! video/x-raw, "
		" width=(int)" + std::to_string(width) + ", "
		" height=(int)" + std::to_string(height) + ", "
		" framerate=(fraction)" + std::to_string(framerate) + ", "
		" videoconvert ! videoscale !"
		" video/x-raw,"
		" width=(int)" + std::to_string(dis_width) + ", "
		" height=(int)" + std::to_string(dis_height) + ", "
		" ! appsink sync=false";
}

int main(int argc, char** argv) {

	std::string pipeline = form_pipeline(320, 320, 50, 320, 320);
	cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

	// open the default camera, use something different from 0 otherwise;
	// Check VideoCapture documentation.
	if(!cap.isOpened()) {
	  return 0;
	}
	
	//cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
	
	cv::Mat image;
	cv::namedWindow("Camera", cv::WINDOW_GUI_NORMAL);
	cv::resizeWindow("Camera", 320, 320);
	
	
	std::chrono::steady_clock::time_point begin;
	std::chrono::steady_clock::time_point end;
	std::chrono::duration<double, std::milli> diff;
	
	std::puts("starting loop");
	std::printf("\n");
	
	while(true) {
		begin = std::chrono::steady_clock::now();
		
		if (!cap.read(image)){
			std::printf("image empty, ending program\n");
			break;
		}
		
		cv::imshow("Camera", image);
		
		end = std::chrono::steady_clock::now();
		diff = end - begin;
		
		std::printf("%f\n", 1000/diff.count());
		//std::printf("\r%.3f fps", 1000/(std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()));
		
		char esc = cv::waitKey(5);
		if(esc == 27) { break; }
	}
	std::printf("\n");
	// the camera will be closed automatically upon exit
	cap.release();
	
	return 0;
}