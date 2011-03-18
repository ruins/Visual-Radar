#include "libfreenect.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class Mutex {
public:
	Mutex() {
		pthread_mutex_init( &m_mutex, NULL );
	}
	void lock() {
		pthread_mutex_lock( &m_mutex );
	}
	void unlock() {
		pthread_mutex_unlock( &m_mutex );
	}
private:
	pthread_mutex_t m_mutex;
};

class MyFreenectDevice : public Freenect::FreenectDevice {
  public:
	MyFreenectDevice(freenect_context *_ctx, int _index)
		: Freenect::FreenectDevice(_ctx, _index), m_buffer_depth(FREENECT_DEPTH_11BIT_SIZE),m_buffer_rgb(FREENECT_VIDEO_RGB_SIZE), m_gamma(2048), m_new_rgb_frame(false), m_new_depth_frame(false),
		  depthMat(Size(640,480),CV_16UC1), rgbMat(Size(640,480),CV_8UC3,Scalar(0)), ownMat(Size(640,480),CV_8UC3,Scalar(0))
	{
		for( unsigned int i = 0 ; i < 2048 ; i++) {
			float v = i/2048.0;
			v = std::pow(v, 3)* 6;
			m_gamma[i] = v*6*256;
		}
	}
	// Do not call directly even in child
	void VideoCallback(void* _rgb, uint32_t timestamp) {
		//std::cout << "RGB callback" << std::endl;
		m_rgb_mutex.lock();
		uint8_t* rgb = static_cast<uint8_t*>(_rgb);
		rgbMat.data = rgb;
		m_new_rgb_frame = true;
		m_rgb_mutex.unlock();
	};
	// Do not call directly even in child
	void DepthCallback(void* _depth, uint32_t timestamp) {
		//std::cout << "Depth callback" << std::endl;
		m_depth_mutex.lock();
		uint16_t* depth = static_cast<uint16_t*>(_depth);
		depthMat.data = (uchar*) depth;
		m_new_depth_frame = true;
		m_depth_mutex.unlock();
	}

	bool getVideo(Mat& output) {
		m_rgb_mutex.lock();
		if(m_new_rgb_frame) {
			cv::cvtColor(rgbMat, output, CV_RGB2BGR);
			m_new_rgb_frame = false;
			m_rgb_mutex.unlock();
			return true;
		} else {
			m_rgb_mutex.unlock();
			return false;
		}
	}

	bool getDepth(Mat& output) {
			m_depth_mutex.lock();
			if(m_new_depth_frame) {
				depthMat.copyTo(output);
				m_new_depth_frame = false;
				m_depth_mutex.unlock();
				return true;
			} else {
				m_depth_mutex.unlock();
				return false;
			}
		}

  private:
	std::vector<uint8_t> m_buffer_depth;
	std::vector<uint8_t> m_buffer_rgb;
	std::vector<uint16_t> m_gamma;
	Mat depthMat;
	Mat rgbMat;
	Mat ownMat;
	Mutex m_rgb_mutex;
	Mutex m_depth_mutex;
	bool m_new_rgb_frame;
	bool m_new_depth_frame;
};

void help()
{
    // print a welcome message, and the OpenCV version
    cout << "\nDriving the Kinect using OpenKinect libraries and,\n"
    		"OpenCV version %s\n" << CV_VERSION << "\n"
    		<< endl;

    cout << "\nHot keys: \n"
            "\tESC - quit the program\n"
            "\Backspace - Take a snapshot\n" << endl;
}

float rawDepth2Metric(int raw_depth)
{
  if (raw_depth < 2047)
  {
   return 1.0 / ((double)raw_depth * (-0.0030711016) + 3.3309495161);
  }
  return 0;
}

void convert2text(string rgbfilename, string depthfilename, string suffix, int iter)
{
	/*******************************************************************************/
	//WHL's Kinect Calibration Parameters
	float fx_rgb =  5.1746500316848039e+02;
	float fy_rgb = 5.1685558974326500e+02;
	float cx_rgb = 3.1163248612153495e+02;
	float cy_rgb = 2.4952548686170906e+02;
	float k1_rgb = 1.7775633376597988e-01;
	float k2_rgb = -6.4039041847027867e-01;
	float p1_rgb = -2.1514204315203123e-02;
	float p2_rgb = -6.8578111533596246e-03;
	float k3_rgb = 7.2369332903167982e-01;

	float fx_d =  5.8027300299853744e+02;
	float fy_d = 5.7785883597877910e+02;
	float cx_d = 3.0609919299427543e+02;
	float cy_d = 2.1034875028398574e+02;
	float k1_d = -9.0109992500784324e-03;
	float k2_d = -7.9860171576037553e-01;
	float p1_d = -1.9406634487223781e-02;
	float p2_d = -5.6568434923388077e-03;
	float k3_d = 2.8702548328343922e+00 ;

	Mat R = (Mat_<float>(3, 3) <<
			9.9997674368389888e-01, -6.7773343679904605e-03,
		   -7.6146583008748173e-04, 6.7799228657314976e-03,
		   9.9997106526615431e-01, 3.4498226631537645e-03,
		   7.3806319557789813e-04, -3.4549051125802996e-03,
		   9.9999375942721880e-01);

	Mat T = (Mat_<float>(3, 1) <<
			2.5220493082789341e-02, 1.0200983335207933e-03,
			       1.6622112503394668e-03);
	/*******************************************************************************/

	for(int i=0;i<iter;i++)
	{
		//saving the RGBD file
		std:: ostringstream rgbdfile;
		rgbdfile << "Data/Textfile/rgbd" << i << ".ply";
		FILE* rgbdFILE = fopen(rgbdfile.str().c_str(), "w+");

		//loading the RGB image
		std::ostringstream rgbfile;
		rgbfile << rgbfilename << i << suffix;
		Mat rgb = cv::imread(rgbfile.str(),1);

		//loading the depth image
		std::ostringstream depthfile;
		depthfile << depthfilename << i << suffix;
		Mat depth = cv::imread(depthfile.str(),-1);
		//cout << depthfile.str()<< endl;

		cout << "image " << i << endl;
		int count = 0;
 		//going through both images
		vector<Scalar> color;
		vector<Mat> coordinates;
		for(int j=0;j<rgb.rows;j++)
		{
			for(int k=0;k<rgb.cols;k++)
			{
				Scalar PColor;
				Mat P3D(3,1,CV_32F);
				Mat temp(3,1,CV_32F);
				Mat P3D2(3,1,CV_32F);
				int rgb_x, rgb_y;
				ushort depthValue = (float)depth.at<ushort>(j,k);
				if(depthValue < 2047)
				{
					float metricDepth = rawDepth2Metric(depthValue);
					P3D.at<float>(0,0) = ((float)k - cx_d)*metricDepth/fx_d;
					P3D.at<float>(1,0) = (((float)j - cy_d)*metricDepth/fy_d);
					P3D.at<float>(2,0) = (metricDepth);

					cv::gemm(R,P3D,1.0,T,1.0,P3D2,0);
					rgb_x = (P3D2.at<float>(0,0)*fx_rgb/P3D2.at<float>(2,0)) + cx_rgb;
					rgb_y = (P3D2.at<float>(1,0)*fy_rgb/P3D2.at<float>(2,0)) + cy_rgb;
					PColor.val[0] = rgb.ptr<uchar>(rgb_y)[3*rgb_x+0];
					PColor.val[1] = rgb.ptr<uchar>(rgb_y)[3*rgb_x+1];
					PColor.val[2] = rgb.ptr<uchar>(rgb_y)[3*rgb_x+2];
					color.push_back(PColor);
					//for display purposes in meshlab
					P3D.at<float>(1,0) = -P3D.at<float>(1,0);
					P3D.at<float>(2,0) = -P3D.at<float>(2,0);
					coordinates.push_back(P3D);
					count++;
				}
			}
		}

		//headers required for '.ply' file
		fprintf(rgbdFILE, "%s\n", "ply");
		fprintf(rgbdFILE, "%s\n", "format ascii 1.0");
		fprintf(rgbdFILE, "%s ", "element vertex");
		fprintf(rgbdFILE, "%d\n", count);
		fprintf(rgbdFILE, "%s\n", "property float x");
		fprintf(rgbdFILE, "%s\n", "property float y");
		fprintf(rgbdFILE, "%s\n", "property float z");
		fprintf(rgbdFILE, "%s\n", "property uchar red");
		fprintf(rgbdFILE, "%s\n", "property uchar green");
		fprintf(rgbdFILE, "%s\n", "property uchar blue");
		fprintf(rgbdFILE, "%s\n", "end_header");

		//writing to file
		for(int i=0;i<count;i++)
		{
			fprintf(rgbdFILE, "%f %f %f %u %u %u\n", coordinates[i].at<float>(0,0), coordinates[i].at<float>(1,0), coordinates[i].at<float>(2,0),
								(uchar)color[i].val[2], (uchar)color[i].val[1], (uchar)color[i].val[0]);
		}

		fclose(rgbdFILE);
	}
}


int main(int argc, char **argv) {
	help();
	bool die(false);
	string filename("snapshot");
	string rgbfilename("Data/RGB/rgb");
	string depthfilename("Data/Depth/depth");
	string suffix(".png");
	int i_snap(0),iter(0);

	Mat depthMat(Size(640,480),CV_16UC1);
	Mat depthf  (Size(640,480),CV_8UC1);
	Mat rgbMat(Size(640,480),CV_8UC3,Scalar(0));
	Mat ownMat(Size(640,480),CV_8UC3,Scalar(0));

	Freenect::Freenect<MyFreenectDevice> freenect;
	MyFreenectDevice& device = freenect.createDevice(0);

	namedWindow("rgb",CV_GUI_NORMAL | CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	namedWindow("depth",CV_GUI_NORMAL | CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	device.startVideo();
	device.startDepth();
    while (!die) {
    	device.getVideo(rgbMat);
    	device.getDepth(depthMat);
        cv::imshow("rgb", rgbMat);
    	depthMat.convertTo(depthf, CV_8UC1, 255.0/2048.0);
        cv::imshow("depth",depthf);
		char k = cvWaitKey(5);
		if( k == 27 ){
		    cvDestroyWindow("rgb");
		    cvDestroyWindow("depth");
			break;
		}
		if( k == 8 ) {
			std::ostringstream file;
			file << filename << i_snap << suffix;
			cv::imwrite(file.str(),rgbMat);
			i_snap++;
		}

		if(iter!=0){
		std::ostringstream rgbfile;
		rgbfile << rgbfilename << iter-1 << suffix;
		cv::imwrite(rgbfile.str(),rgbMat);

		std::ostringstream depthfile;
		depthfile << depthfilename << iter-1 << suffix;
		cv::imwrite(depthfile.str(),depthMat);
		}

		iter++;
    }

   	device.stopVideo();
	device.stopDepth();

	//converting the data to textfile
	convert2text(rgbfilename, depthfilename, suffix, iter-1);

	return 0;
}
