#include "features_matcher.h"
#include <iostream>
#include <map>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

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
        int minHessian = 400;
        cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
        cv::Mat descriptor;
        detector->detectAndCompute( img, cv::noArray(), features_[i], descriptor );

        for(int k=0; k<features_[i].size(); k++) {
            cv::Vec3b color = img.at<cv::Vec3b>(features_[i][k].pt);
            feats_colors_[i].push_back(color);
        }


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
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );
    for( int i = 0; i < images_names_.size() - 1; i++ )
    {
        for( int j = i + 1; j < images_names_.size(); j++ )
        {
            cv:: Mat img1 = readUndistortedImage(images_names_[i]);
            cv:: Mat img2 = readUndistortedImage(images_names_[j]);
            std::cout<<"Matching image "<<i<<" with image "<<j<<std::endl;
            std::vector<cv::DMatch> matches, inlier_matches;
            //BFMatcher matcher ( NORM_HAMMING );
            matcher->match ( descriptors_[i], descriptors_[j], matches );
            for (int p = 0; p < matches.size(); ++p)
            {
                const float ratio = 0.4; // As in Lowe's paper; can be tuned
                if (matches[p].distance < ratio * matches[p+1].distance)
                {
                    inlier_matches.push_back(matches[p]);
                }
            }

            if(inlier_matches.size()<=10)
                break;

//            cv::Mat img_matches;
//            cv::drawMatches( img1, features_[i], img2, features_[j], inlier_matches, img_matches, cv::Scalar::all(-1),
//                         cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//            //-- Show detected matches
//            imshow("Good Matches", img_matches );
//            cv::waitKey(0);
            intrinsics_matrix_ = new_intrinsics_matrix_;

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
        cv::Mat src_obs( num_observations_,1, cv :: traits :: Type <cv :: Vec2d> :: value, const_cast<double *>(observations_.data()));
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