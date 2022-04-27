#pragma once

#pragma once

#include <string>
#include <vector>

#include "Eigen/Dense"
#include <opencv2/opencv.hpp>

class BasicSfM
{
 public:

  ~BasicSfM();

  // Read data from file, that are observations along with the ids of the camera positions
  // where the observations have been acquired and the ids of the 3D points that generate such observations)
  // If load_initial_guess is set to true, it is assumed that the input file also includes an initial guess
  // solution of the reconstruction problem, hence load it
  // If load_colors is set to true, it is assumed that the input file also includes the RGB colors of the
  // 3D points, hence load them
  void readFromFile(const std::string& filename, bool load_initial_guess = false, bool load_colors = false );
  // Write data to file, if write_unoptimized is set to true, also the optimized parameters (camera and points
  // positions) are stored to file
  void writeToFile (const std::string& filename, bool write_unoptimized = false  ) const;

  // Save the reconstructed scene in a PLY file: you can visualize it by using for instance MeshLab
  // (execute
  // sudo apt install meshlab
  // in debian-derived linux distributions to install meshlab)
  void writeToPLYFile (const std::string& filename, bool write_unoptimized = false ) const;

  // The core of this class: it performs incremental structure from motion on the loaded data
  void solve();

  // Clear everything
  void reset();

 private:

  // Refine camera and point position registered so far inside a global optimization problem
  void bundleAdjustmentIter( int new_cam_idx );

  // A simple strategy for eliminating outliers: just check the projection error of each point in each view, if is is greater than max_reproj_err_, remove the point from the solution
  int rejectOuliers();

  // Get the pointer to the 6-dimensional parameter block that defines the position of the pose_idx-th view
  inline double *cameraBlockPtr ( int pose_idx = 0 ) const
  {
    return const_cast<double *>(parameters_.data()) + (pose_idx * camera_block_size_ );
  };

  // Get the pointer to the 3-dimensional parameter block that defines the position of the point_idx-th point
  inline double *pointBlockPtr ( int point_idx = 0 ) const
  {
    return const_cast<double *>(parameters_.data()) + (num_poses_ * camera_block_size_ + point_idx * point_block_size_ );
  };

  // Initialize the point_idx-th point parameter as seen at depth meters from the pos_idx position
  void initPointParams(int pt_idx, int pos_idx, const double img_p[2], double depth);
  void initCamParams(int new_pose_idx, cv::Mat r_vec, cv::Mat t_vec );

  void cam2center (const double* camera, double* center) const;
  void center2cam (const double* center, double* camera) const;

  // Test the cheirality constaint for the pt_idx-th poiunt seen by the pos_idx-th camera
  bool checkCheiralityConstraint(int pos_idx, int pt_idx );

  // Print the the 6-dimensional parameter block that defines the position of the idx-th view
  void printPose( int idx ) const;

  // Print the the 3-dimensional parameter block that defines the position of the idx-th point
  void printPointParams( int idx ) const;

  int num_poses_ = 0;
  int num_points_ = 0;
  int num_observations_ = 0;
  int num_parameters_ = 0;

  // For each observation (i.e., projection into an image plane) store the *index* of the corresponding 3D point
  // that generates the observation. point_index_ has a size equal to num_observations_
  std::vector<int> point_index_;
  // For each observation (i.e., projection into an image plane) store the *index* of the corresponding 6-DoF position
  // of the camera that made the observation. pose_index_ has a size equal to num_observations_
  std::vector<int> pose_index_;
  // Vector of observation, i.e. 2D point projections. observations_ has a size equal to 2*num_observations_
  std::vector<double> observations_;
  // Vector of the colors of the observed 3D points (if available). colors_ has a size equal to 3*num_points_
  std::vector<unsigned char> colors_;

  // Vector of all the paramters to be estimated: it is composed by num_poses_ 6D R,t blocks (one for each view)
  // followed by num_points_ 3D blocks (one for each point)
  std::vector<double> parameters_;

  const int camera_block_size_ = 6;
  const int point_block_size_ = 3;

  // For each pose, the number of optimization iterations (0 if it has not yet been estimated,
  // -1 if the pose has been rejected)
  std::vector<int> pose_optim_iter_;
  // For each point, the number of optimization iterations (0 if it has not yet been estimated,
  // -1 if the point has been rejected)
  std::vector<int> pts_optim_iter_;

  // Maximum reprojection error used to classify an inlier
  double max_reproj_err_ = 0.01;
  // Maximum number of outliers that we can tolerate without re-optimizing all
  int max_outliers_ = 5;
};
