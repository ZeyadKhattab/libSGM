/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/version.hpp>

#include <libsgm.h>

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

struct device_buffer
{
	device_buffer() : data(nullptr) {}
	device_buffer(size_t count) { allocate(count); }
	void allocate(size_t count) { cudaMalloc(&data, count); }
	~device_buffer() { cudaFree(data); }
	void* data;
};

template <class... Args>
static std::string format_string(const char* fmt, Args... args)
{
	const int BUF_SIZE = 1024;
	char buf[BUF_SIZE];
	std::snprintf(buf, BUF_SIZE, fmt, args...);
	return std::string(buf);
}
std::string padNum(int x){
	std::string ans=std::to_string(x);
	while(ans.length()<6)
	ans="0"+ans;
	return ans;
}
int main(int argc, char* argv[])
{
	if (argc < 3) {
		std::cout << "usage: " << argv[0] << " left-image-format right-image-format [disp_size]" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	const int first_frame = 0;

	cv::Mat I1 = cv::imread(format_string(argv[1], first_frame), -1);
	cv::Mat I2 = cv::imread(format_string(argv[2], first_frame), -1);
	const int disp_size = argc >= 4 ? std::stoi(argv[3]) : 128;

	ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed.");
	ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
	ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(disp_size == 64 || disp_size == 128 || disp_size == 256, "disparity size must be 64, 128 or 256.");

	const int width = I1.cols;
	const int height = I1.rows;

	const int input_depth = I1.type() == CV_8U ? 8 : 16;
	const int input_bytes = input_depth * width * height / 8;
	const int output_depth = disp_size < 256 ? 8 : 16;
	const int output_bytes = output_depth * width * height / 8;

	sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);

	const int invalid_disp = output_depth == 8
			? static_cast< uint8_t>(sgm.get_invalid_disparity())
			: static_cast<uint16_t>(sgm.get_invalid_disparity());

	cv::Mat disparity(height, width, output_depth == 8 ? CV_8U : CV_16U);
	cv::Mat disparity_8u, disparity_color;

	device_buffer d_I1(input_bytes), d_I2(input_bytes), d_disparity(output_bytes);

	for (int frame_no = first_frame;frame_no<200; frame_no++) {

		I1 = cv::imread(format_string(argv[1], frame_no), -1);
		I2 = cv::imread(format_string(argv[2], frame_no), -1);
	/*	if (I1.empty() || I2.empty()) {
			frame_no = first_frame;
			continue;
		}*/

		cudaMemcpy(d_I1.data, I1.data, input_bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_I2.data, I2.data, input_bytes, cudaMemcpyHostToDevice);

		

		sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
		

	

		cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);
		disparity.setTo(0, disparity > (int32_t)49152);
		disparity *= 256;






		cv::imwrite("disp_0/"+padNum(frame_no)+"_10.png", disparity);
	}

	return 0;
}
