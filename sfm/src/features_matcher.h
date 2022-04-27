#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class FeatureMatcher
{
 public:

  // Constructor: it require the camera intrinsics matrix, its distortion coefficients and an optional
  // focal length scaling factor
  FeatureMatcher( cv::Mat intrinsics_matrix, cv::Mat dist_coeffs, double focal_scale = 1.0 );

  // Set the list of names of the images from which to extract the features
  void setImagesNames( const std::vector<std::string> &images_names ) { images_names_ = images_names; };

  // Extract from each image a set of salient points and compute descriptors
  void extractFeatures();

  // Perform exhaustive matching between features descriptors, and internally store the results
  void exhaustiveMatching();

  // Write the results to file. If normalize_points is true, normalize observations as seen by the canonical camera
  void writeToFile ( const std::string& filename, bool normalize_points ) const;

  // Show matches collecting data  (rescale images with scale)
  void testMatches( double scale = 1.0 );

  // Clear everything
  void reset();

 private:

  // Read from file an image and undistort it, possibly by rescaling the focal length
  // (see the focal_scale parameter of the constructor)
  cv::Mat readUndistortedImage(const std::string& filename );

  // Get a single, unique ID from a pair [position ID, feature ID] (used as hash)
  uint64_t poseFeatPairID( int pose_id, int feat_id )
  {
    return static_cast<uint64_t>(pose_id) | static_cast<uint64_t>(feat_id)<<32;
  };

  // Add the matches between two images
  void setMatches( int pos0_id, int pos1_id, const std::vector<cv::DMatch> &matches );

  cv::Mat intrinsics_matrix_, dist_coeffs_, new_intrinsics_matrix_;

  std::unordered_map<int, int> pose_id_map_;
  std::unordered_map<uint64_t, int> point_id_map_;

  std::vector<std::string> images_names_;
  std::vector< std::vector<cv::KeyPoint> > features_;
  std::vector< std::vector<cv::Vec3b > > feats_colors_;
  std::vector< cv::Mat > descriptors_;


  int num_poses_ = 0;
  int num_points_ = 0;
  int num_observations_ = 0;

  std::vector<int> point_index_;
  std::vector<int> pose_index_;
  std::vector<double> observations_;
  std::vector<unsigned char> colors_;
};