#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

bool readFileNamesFromFolder ( const std::string& input_folder_name, std::vector< std::string >& names );
bool loadCameraParams( const std::string &file_name, cv::Size &image_size,
                       cv::Mat &camera_matrix, cv::Mat &dist_coeffs );
