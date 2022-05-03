
#include "features_matcher.h"

#include <iostream>
#include <map>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include"opencv2/imgproc.hpp"
#include "iostream"


using namespace cv::xfeatures2d;
using namespace cv;

const double FOCAL_LENGTH = 4308 ; // focal length in pixels, after downsampling, guess from jpeg EXIF data



namespace
{
template<typename T>
void FscanfOrDie(FILE *fptr, const char *format, T *value)
{
  int num_scanned = fscanf(fptr, format, value);
  if (num_scanned != 1)
  {
    std::cerr << "Invalid UW data file.";
    exit(-1);
  }
}
}

FeatureMatcher::FeatureMatcher(cv::Mat intrinsics_matrix, cv::Mat dist_coeffs, double focal_scale)
{
  intrinsics_matrix_ = intrinsics_matrix.clone();
  dist_coeffs_ = dist_coeffs.clone();
  new_intrinsics_matrix_ = intrinsics_matrix.clone();
  new_intrinsics_matrix_.at<double>(0,0) *= focal_scale;
  new_intrinsics_matrix_.at<double>(1,1) *= focal_scale;
}

cv::Mat FeatureMatcher::readUndistortedImage(const std::string& filename )
{
  cv::Mat img = cv::imread(filename), und_img, dbg_img;
  cv::undistort	(	img, und_img, intrinsics_matrix_, dist_coeffs_, new_intrinsics_matrix_ );

  return und_img;
}

void FeatureMatcher::extractFeatures()
{

  features_.resize(images_names_.size());
  descriptors_.resize(images_names_.size());
  feats_colors_.resize(images_names_.size());

  for( int i = 0; i < images_names_.size(); i++  )

  {
    std::cout<<"Computing descriptors for image "<<i<<std::endl;
    cv::Mat img = readUndistortedImage(images_names_[i]);

      std::vector< std::vector<cv::Vec3b > > feats_colors;

      std::vector< std::vector<cv::KeyPoint> > features;

      std::vector< cv::Mat > descriptors;

      Ptr<FeatureDetector> detector = ORB::create();
      Ptr<DescriptorExtractor> descriptor = ORB::create();

      Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );


      detector->detect ( img,features_[i]);

      descriptor->compute (img, features_[i], descriptors_[i]);


      Mat outimg_1;
      cv::drawKeypoints( img, features_[i], outimg_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
      imshow("ORB",outimg_1);
      waitKey(0);




      //////////////////////////// Code to be completed (1/1) /////////////////////////////////
    // Extract salient points + descriptors from i-th image, and store them into
    // features_[i] and descriptors_[i] vector, respectively
    // Extract also the color (i.e., the cv::Vec3b information) of each feature, and store
    // it into feats_colors_[i] vector
    /////////////////////////////////////////////////////////////////////////////////////////
  }
}

void FeatureMatcher::exhaustiveMatching()
{
  for( int i = 0; i < images_names_.size() - 1; i++ )
  {
    for( int j = i + 1; j < images_names_.size(); j++ )
    {
      std::cout<<"Matching image "<<i<<" with image "<<j<<std::endl;
      std::vector<cv::DMatch> matches, inlier_matches;

        cv::Mat img_1 = readUndistortedImage(images_names_[i]);
        cv::Mat img_2 = readUndistortedImage(images_names_[j]);


//
////        descriptor->compute (img, features_[i], feats_colors_[i]);
//
        Ptr<FeatureDetector> detector = ORB::create();
        Ptr<DescriptorExtractor> descriptor = ORB::create();

//      Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );

        matcher->match ( descriptors_[i], descriptors_[j], matches);

//

        Mat img_match,img_match_1;
        drawMatches ( img_1, features_[i], img_2, features_[j], matches, img_match );
        imshow ( "img_match", img_match );
        waitKey(0);

//       drawMatches ( img_1, features_[i], img_2, features_[j], inlier_matches, img_match_1 );
//       imshow ( "img_match_1", img_match_1 );
//       waitKey(0);

        std::vector< DMatch > good_matches;
//        double min_dist=50, max_dist=0;
//
//        for ( int p = 0; p < descriptors_[p].rows; p++ )
//        {
//            std:: cout <<matches[p].distance<<std::endl;
//            std:: cout <<max ( min_dist, 75.0 )<<std::endl;
//            std:: cout <<"*******************"<<std::endl;
//            if ( matches[p].distance <= max ( min_dist, 70.0 ) )
//            {
//                good_matches.push_back (matches[p]);
//            }
//        }
        for (int p = 0; p < matches.size()-1; p++)
        {
            const float ratio = 0.4; // As in Lowe's paper; can be tuned
            if (matches[p].distance < ratio * matches[p+1].distance)
            {
                good_matches.push_back(matches[p]);
            }
        }
        std::cout<<good_matches.size()<<std::endl;

        if(good_matches.size()<=10){
            break;
        }
        Mat x;
        std::cout<<i<<j<<std::endl;
        std::cout<<"****************"<<std::endl;

        drawMatches ( img_1, features_[i], img_2, features_[j], good_matches, x );
        imshow ( "x", x );
        waitKey(0);




//        // Convert keypoints into Point2f
        std::vector<cv::Point2f> points1,points2;
//
        for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
            // Get the position of left keypoints

            float x = features_[i][it->queryIdx].pt.x;
            float y = features_[i][it->queryIdx].pt.y;

            points1.push_back(cv::Point2f(x, y));


            // Get the position of right keypoints
            x = features_[j][it->trainIdx].pt.x;
            y = features_[j][it->trainIdx].pt.y;
            points2.emplace_back(x, y);
        }
//
//        // Find the essential between image 1 and image 2
        cv::Mat inliers;

        cv::Mat essential = cv::findEssentialMat(points1, points2, new_intrinsics_matrix_, cv::RANSAC, 0.9, 1.0, inliers);
        std::cout<<essential<< std::endl;
//
////        // recover relative camera pose from essential matrix
        cv::Mat rotation, translation;
        cv::recoverPose(essential, points1, points2, new_intrinsics_matrix_, rotation, translation, inliers);
        std::cout<<rotation<<std::endl;
        std::cout<<translation<<std::endl;
////
        // compose projection matrix from R,T
        cv::Mat projection2(3, 4, CV_64F); // the 3x4 projection matrix
        rotation.copyTo(projection2(cv::Rect(0, 0, 3, 3)));
        translation.copyTo(projection2.colRange(3, 4));
////
//        // compose generic projection matrix
        cv::Mat projection1(3, 4, CV_64F, 0.); // the 3x4 projection matrix
        cv::Mat diag(cv::Mat::eye(3, 3, CV_64F));
        diag.copyTo(projection1(cv::Rect(0, 0, 3, 3)));
////
//        // to contain the inliers
        std::vector<cv::Vec2d> inlierPts1;
        std::vector<cv::Vec2d> inlierPts2;
////
////        // create inliers input point vector for triangulation
        int t(0);
        for (int l = 0; l < inliers.rows; l++) {
            if (inliers.at<uchar>(l)) {
                inlierPts1.push_back(cv::Vec2d(points1[l].x, points1[l].y));
                inlierPts2.push_back(cv::Vec2d(points2[l].x, points2[l].y));
            }
        }
////        // undistort and normalize the image points
        std::vector<cv::Vec2d> points1u;
        cv::undistortPoints(inlierPts1, points1u, new_intrinsics_matrix_, dist_coeffs_);
        std::vector<cv::Vec2d> points2u;
        cv::undistortPoints(inlierPts2, points2u, new_intrinsics_matrix_, dist_coeffs_);
////
////        // Triangulation
////        std::vector<cv::Vec3d> points3D;
////        triangulatePoints(projection1, projection2, points1u, points2u, points3D);
////
////        std::cout<<"3D points :"<<points3D.size()<<std::endl;
//
//        //-- Localize the object
////        std::vector<Point2f> obj;
////        std::vector<Point2f> scene;
////        std::vector<int> i_kp, j_kp;
////        std::vector<uchar> mask;
////
////
//        Mat H = findHomography( points1u, points2u, RANSAC );
////
////
////        //-- Get the corners from the image_1 ( the object to be "detected" )
//        std::vector<Point2f> obj_corners(4);
//        obj_corners[0] = Point2f(0, 0);
//        obj_corners[1] = Point2f( (float)img_1.cols, 0 );
//        obj_corners[2] = Point2f( (float)img_1.cols, (float)img_1.rows );
//        obj_corners[3] = Point2f( 0, (float)img_1.rows );
//        std::vector<Point2f> scene_corners(4);
//        perspectiveTransform( obj_corners, scene_corners, H);
//
//        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
//        line( img_match, scene_corners[0] + Point2f((float)img_1.cols, 0),
//              scene_corners[1] + Point2f((float)img_1.cols, 0), Scalar(0, 255, 0), 4 );
//        line( img_match, scene_corners[1] + Point2f((float)img_1.cols, 0),
//              scene_corners[2] + Point2f((float)img_1.cols, 0), Scalar( 0, 255, 0), 4 );
//        line( img_match, scene_corners[2] + Point2f((float)img_1.cols, 0),
//              scene_corners[3] + Point2f((float)img_1.cols, 0), Scalar( 0, 255, 0), 4 );
//        line( img_match, scene_corners[3] + Point2f((float)img_1.cols, 0),
//              scene_corners[0] + Point2f((float)img_1.cols, 0), Scalar( 0, 255, 0), 4 );
//
////
//        //-- Show detected matches
//        imshow("Good Matches & Object detection", img_match );
//        waitKey(0);



////

        //////////////////////////// Code to be completed (2/5) /////////////////////////////////
      // Match descriptors between image i and image j, and perform geometric validation,
      // possibly discarding the outliers (remember that features have been extracted
      // from undistorted images that now has  new_intrinsics_matrix_ as K matrix and
      // no distortions)
      // As geometric models, use both the Essential matrix and the Homograph matrix,
      // both by setting new_intrinsics_matrix_ as K matrix
      // Do not set matches between two images if the amount of inliers matches
      // (i.e., geomatrically verified matches) is small (say <= 10 matches)
      /////////////////////////////////////////////////////////////////////////////////////////

      setMatches( i, j, inlier_matches);

    }
  }
}

void FeatureMatcher::writeToFile ( const std::string& filename, bool normalize_points ) const
{
  FILE* fptr = fopen(filename.c_str(), "w");

  if (fptr == NULL) {
    std::cerr << "Error: unable to open file " << filename;
    return;
  };

  fprintf(fptr, "%d %d %d\n", num_poses_, num_points_, num_observations_);

  double *tmp_observations;
  cv::Mat dst_pts;
  if(normalize_points)
  {
    cv::Mat src_obs( num_observations_,1, cv :: traits :: Type <cv :: Vec2d> :: value
            , const_cast<double *>(observations_.data()));
    cv::undistortPoints(src_obs, dst_pts, new_intrinsics_matrix_, cv::Mat());
    tmp_observations = reinterpret_cast<double *>(dst_pts.data);
  }
  else
  {
    tmp_observations = const_cast<double *>(observations_.data());
  }

  for (int i = 0; i < num_observations_; ++i)
  {
    fprintf(fptr, "%d %d", pose_index_[i], point_index_[i]);
    for (int j = 0; j < 2; ++j) {
      fprintf(fptr, " %g", tmp_observations[2 * i + j]);
    }
    fprintf(fptr, "\n");
  }

  if( colors_.size() == 3*num_points_ )
  {
    for (int i = 0; i < num_points_; ++i)
      fprintf(fptr, "%d %d %d\n", colors_[i*3], colors_[i*3 + 1], colors_[i*3 + 2]);
  }

  fclose(fptr);
}

void FeatureMatcher::testMatches( double scale )
{
  // For each pose, prepare a map that reports the pairs [point index, observation index]
  std::vector< std::map<int,int> > cam_observation( num_poses_ );
  for( int i_obs = 0; i_obs < num_observations_; i_obs++ )
  {
    int i_cam = pose_index_[i_obs], i_pt = point_index_[i_obs];
    cam_observation[i_cam][i_pt] = i_obs;
  }

  for( int r = 0; r < num_poses_; r++ )
  {
    for (int c = r + 1; c < num_poses_; c++)
    {
      int num_mathces = 0;
      std::vector<cv::DMatch> matches;
      std::vector<cv::KeyPoint> features0, features1;
      for (auto const &co_iter: cam_observation[r])
      {
        if (cam_observation[c].find(co_iter.first) != cam_observation[c].end())
        {
          features0.emplace_back(observations_[2*co_iter.second],observations_[2*co_iter.second + 1], 0.0);
          features1.emplace_back(observations_[2*cam_observation[c][co_iter.first]],observations_[2*cam_observation[c][co_iter.first] + 1], 0.0);
          matches.emplace_back(num_mathces,num_mathces, 0);
          num_mathces++;
        }
      }
      cv::Mat img0 = readUndistortedImage(images_names_[r]),
          img1 = readUndistortedImage(images_names_[c]),
          dbg_img;

      cv::drawMatches(img0, features0, img1, features1, matches, dbg_img);
      cv::resize(dbg_img, dbg_img, cv::Size(), scale, scale);
      cv::imshow("", dbg_img);
      if (cv::waitKey() == 27)
        return;
    }
  }
}

void FeatureMatcher::setMatches( int pos0_id, int pos1_id, const std::vector<cv::DMatch> &matches )
{

  const auto &features0 = features_[pos0_id];
  const auto &features1 = features_[pos1_id];

  auto pos_iter0 = pose_id_map_.find(pos0_id),
      pos_iter1 = pose_id_map_.find(pos1_id);

  // Already included position?
  if( pos_iter0 == pose_id_map_.end() )
  {
    pose_id_map_[pos0_id] = num_poses_;
    pos0_id = num_poses_++;
  }
  else
    pos0_id = pose_id_map_[pos0_id];

  // Already included position?
  if( pos_iter1 == pose_id_map_.end() )
  {
    pose_id_map_[pos1_id] = num_poses_;
    pos1_id = num_poses_++;
  }
  else
    pos1_id = pose_id_map_[pos1_id];

  for( auto &match:matches)
  {

    // Already included observations?
    uint64_t obs_id0 = poseFeatPairID(pos0_id, match.queryIdx ),
        obs_id1 = poseFeatPairID(pos1_id, match.trainIdx );
    auto pt_iter0 = point_id_map_.find(obs_id0),
        pt_iter1 = point_id_map_.find(obs_id1);
    // New point
    if( pt_iter0 == point_id_map_.end() && pt_iter1 == point_id_map_.end() )
    {
      int pt_idx = num_points_++;
      point_id_map_[obs_id0] = point_id_map_[obs_id1] = pt_idx;

      point_index_.push_back(pt_idx);
      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos0_id);
      pose_index_.push_back(pos1_id);
      observations_.push_back(features0[match.queryIdx].pt.x);
      observations_.push_back(features0[match.queryIdx].pt.y);
      observations_.push_back(features1[match.trainIdx].pt.x);
      observations_.push_back(features1[match.trainIdx].pt.y);

      // Average color between two corresponding features (suboptimal since we shouls also consider
      // the other observations of the same point in the other images)
      cv::Vec3f color = (cv::Vec3f(feats_colors_[pos0_id][match.queryIdx]) +
                        cv::Vec3f(feats_colors_[pos1_id][match.trainIdx]))/2;

      colors_.push_back(cvRound(color[2]));
      colors_.push_back(cvRound(color[1]));
      colors_.push_back(cvRound(color[0]));

      num_observations_++;
      num_observations_++;
    }
      // New observation
    else if( pt_iter0 == point_id_map_.end() )
    {
      int pt_idx = point_id_map_[obs_id1];
      point_id_map_[obs_id0] = pt_idx;

      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos0_id);
      observations_.push_back(features0[match.queryIdx].pt.x);
      observations_.push_back(features0[match.queryIdx].pt.y);
      num_observations_++;
    }
    else if( pt_iter1 == point_id_map_.end() )
    {
      int pt_idx = point_id_map_[obs_id0];
      point_id_map_[obs_id1] = pt_idx;

      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos1_id);
      observations_.push_back(features1[match.trainIdx].pt.x);
      observations_.push_back(features1[match.trainIdx].pt.y);
      num_observations_++;
    }
//    else if( pt_iter0->second != pt_iter1->second )
//    {
//      std::cerr<<"Shared observations does not share 3D point!"<<std::endl;
//    }
  }
}
void FeatureMatcher::reset()
{
  point_index_.clear();
  pose_index_.clear();
  observations_.clear();
  colors_.clear();

  num_poses_ = num_points_ = num_observations_ = 0;
}