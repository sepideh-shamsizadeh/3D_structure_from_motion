#include "basic_sfm.h"

#include <iostream>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <fstream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;

struct ReprojectionError
{
    ReprojectionError(double observed_x, double observed_y)
            : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = - p[0] / p[2];
        T yp = - p[1] / p[2];

        // Apply second and fourth order radial distortion.
        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = xp*xp + yp*yp;
        T distortion = 1.0 + r2  * (l1 + l2  * r2);

        // Compute final projected point position.
        const T& focal = camera[6];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 9, 3>(
                new ReprojectionError(observed_x, observed_y)));
    }

    double observed_x;
    double observed_y;
    //////////////////////////// Code to be completed (5/5) //////////////////////////////////
    // This class should include an auto-differentiable cost function (see Ceres Solver docs).
    // Remember that we are dealing with a normalized, canonical camera:
    // point projection is easy! To rotete a point given an axis-angle rotation, use
    // the Ceres function:
    // AngleAxisRotatePoint(...) (see ceres/rotation.h)
    // WARNING: When dealing with the AutoDiffCostFunction template parameters,
    // pay attention to the order of the template parameters
    //////////////////////////////////////////////////////////////////////////////////////////
};

/* As ReprojectionError, but the camera parameters are fixed */
struct PointReprojectionError
{
    PointReprojectionError( double cam_rx, double cam_ry, double cam_rz,
                            double cam_x, double cam_y, double cam_z,
                            double observed_x, double observed_y
    )
            : cam_r_vec(cam_rx, cam_ry, cam_rz),
              cam_t_vec(cam_x, cam_y, cam_z),
              observed_x(observed_x), observed_y(observed_y){}

    template <typename T>
    bool operator()(const T* const point,
                    T* residuals) const
    {
        T r_vec[3], p[3];
        r_vec[0] = T(cam_r_vec(0));
        r_vec[1] = T(cam_r_vec(1));
        r_vec[2] = T(cam_r_vec(2));

        ceres::AngleAxisRotatePoint( r_vec, point, p);

        // camera[3,4,5] are the translation.
        p[0] += T(cam_t_vec(0));
        p[1] += T(cam_t_vec(1));
        p[2] += T(cam_t_vec(2));

        T predicted_x = p[0] / p[2];
        T predicted_y = p[1] / p[2];

        // The error is the difference between the predicted and observed position.
        residuals[0] = (predicted_x - observed_x);
        residuals[1] = (predicted_y - observed_y);

        return true;
    }

    static ceres::CostFunction* Create(double cam_rx, double cam_ry, double cam_rz,
                                       double cam_x, double cam_y, double cam_z,
                                       double observed_x, double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<PointReprojectionError, 2, 3>(
                new PointReprojectionError( cam_rx, cam_ry, cam_rz, cam_x, cam_y, cam_z,
                                            observed_x, observed_y)));
    }

    Eigen::Vector3d cam_r_vec;
    Eigen::Vector3d cam_t_vec;

    double observed_x, observed_y;
};

namespace
{
    typedef Eigen::Map<Eigen::VectorXd> VectorRef;
    typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

    template<typename T>
    void FscanfOrDie(FILE* fptr, const char* format, T* value)
    {
        int num_scanned = fscanf(fptr, format, value);
        if (num_scanned != 1)
        {
            cerr << "Invalid UW data file.";
            exit(-1);
        }

    }

}  // namespace


BasicSfM::~BasicSfM()
{
    reset();
}

void BasicSfM::reset()
{
    point_index_.clear();
    pose_index_.clear();
    observations_.clear();
    colors_.clear();
    parameters_.clear();

    num_poses_ = num_points_ = num_observations_ = num_parameters_ = 0;
}

void BasicSfM::readFromFile ( const std::string& filename, bool load_initial_guess, bool load_colors  )
{
    reset();

    FILE* fptr = fopen(filename.c_str(), "r");

    if (fptr == NULL)
    {
        cerr << "Error: unable to open file " << filename;
        return;
    };

    // This wil die horribly on invalid files. Them's the breaks.
    FscanfOrDie(fptr, "%d", &num_poses_);
    FscanfOrDie(fptr, "%d", &num_points_);
    FscanfOrDie(fptr, "%d", &num_observations_);

    cout << "Header: " << num_poses_
         << " " << num_points_
         << " " << num_observations_;

    point_index_.resize(num_observations_);
    pose_index_.resize(num_observations_);
    observations_.resize(2 * num_observations_);

    num_parameters_ = camera_block_size_ * num_poses_ + point_block_size_ * num_points_;
    parameters_.resize(num_parameters_);

    for (int i = 0; i < num_observations_; ++i)
    {
        FscanfOrDie(fptr, "%d", pose_index_.data() + i);
        FscanfOrDie(fptr, "%d", point_index_.data() + i);
        for (int j = 0; j < 2; ++j)
        {
            FscanfOrDie(fptr, "%lf", observations_.data() + 2*i + j);
        }
    }

    if( load_colors )
    {
        colors_.resize(3*num_points_);
        for (int i = 0; i < num_points_; ++i)
        {
            int r,g,b;
            FscanfOrDie(fptr, "%d", &r );
            FscanfOrDie(fptr, "%d", &g);
            FscanfOrDie(fptr, "%d", &b );
            colors_[i*3] = r;
            colors_[i*3 + 1] = g;
            colors_[i*3 + 2] = b;
        }
    }

    if( load_initial_guess )
    {
        pose_optim_iter_.resize( num_poses_, 1 );
        pts_optim_iter_.resize( num_points_, 1 );

        for (int i = 0; i < num_parameters_; ++i)
        {
            FscanfOrDie(fptr, "%lf", parameters_.data() + i);
        }
    }
    else
    {
        memset(parameters_.data(), 0, num_parameters_*sizeof(double));
        // Masks used to indicate which cameras and points have been optimized so far
        pose_optim_iter_.resize( num_poses_, 0 );
        pts_optim_iter_.resize( num_points_, 0 );
    }

    fclose(fptr);
}


void BasicSfM::writeToFile (const string& filename, bool write_unoptimized ) const
{
    FILE* fptr = fopen(filename.c_str(), "w");

    if (fptr == NULL) {
        cerr << "Error: unable to open file " << filename;
        return;
    };

    if( write_unoptimized )
    {
        fprintf(fptr, "%d %d %d\n", num_poses_, num_points_, num_observations_);

        for (int i = 0; i < num_observations_; ++i)
        {
            fprintf(fptr, "%d %d", pose_index_[i], point_index_[i]);
            for (int j = 0; j < 2; ++j) {
                fprintf(fptr, " %g", observations_[2 * i + j]);
            }
            fprintf(fptr, "\n");
        }

        if( colors_.size() == num_points_*3 )
        {
            for (int i = 0; i < num_points_; ++i)
                fprintf(fptr, "%d %d %d\n", colors_[i*3], colors_[i*3 + 1], colors_[i*3 + 2]);
        }

        for (int i = 0; i < num_poses_; ++i)
        {
            const double *camera = parameters_.data() + camera_block_size_ * i;
            for (int j = 0; j < camera_block_size_; ++j) {
                fprintf(fptr, "%.16g\n", camera[j]);
            }
        }

        const double* points = pointBlockPtr();
        for (int i = 0; i < num_points_; ++i)
        {
            const double* point = points + i * point_block_size_;
            for (int j = 0; j < point_block_size_; ++j) {
                fprintf(fptr, "%.16g\n", point[j]);
            }
        }
    }
    else
    {
        int num_cameras = 0, num_points = 0, num_observations = 0;

        for (int i = 0; i < num_poses_; ++i)
            if( pose_optim_iter_[i] > 0 ) num_cameras++;

        for (int i = 0; i < num_points_; ++i)
            if( pts_optim_iter_[i] > 0 ) num_points++;

        for (int i = 0; i < num_observations_; ++i)
            if( pose_optim_iter_[pose_index_[i]] > 0  && pts_optim_iter_[point_index_[i]] > 0 ) num_observations++;

        fprintf(fptr, "%d %d %d\n", num_cameras, num_points, num_observations);

        for (int i = 0; i < num_observations_; ++i)
        {
            if( pose_optim_iter_[pose_index_[i]] > 0  && pts_optim_iter_[point_index_[i]] > 0 )
            {
                fprintf(fptr, "%d %d", pose_index_[i], point_index_[i]);
                for (int j = 0; j < 2; ++j) {
                    fprintf(fptr, " %g", observations_[2 * i + j]);
                }
                fprintf(fptr, "\n");
            }
        }

        if( colors_.size() == num_points_*3 )
        {
            for (int i = 0; i < num_points_; ++i)
            {
                if(pts_optim_iter_[i] > 0)
                    fprintf(fptr, "%d %d %d\n", colors_[i*3], colors_[i*3 + 1], colors_[i*3 + 2]);
            }
        }

        for (int i = 0; i < num_poses_; ++i)
        {
            if( pose_optim_iter_[i] > 0 )
            {
                const double *camera = parameters_.data() + camera_block_size_ * i;
                for (int j = 0; j < camera_block_size_; ++j)
                {
                    fprintf(fptr, "%.16g\n", camera[j]);
                }
            }
        }

        const double* points = pointBlockPtr();
        for (int i = 0; i < num_points_; ++i)
        {
            if( pts_optim_iter_[i] > 0 )
            {
                const double* point = points + i * point_block_size_;
                for (int j = 0; j < point_block_size_; ++j)
                {
                    fprintf(fptr, "%.16g\n", point[j]);
                }
            }
        }
    }

    fclose(fptr);
}

// Write the problem to a PLY file for inspection in Meshlab or CloudCompare.
void BasicSfM::writeToPLYFile (const string& filename, bool write_unoptimized ) const
{
    ofstream of(filename.c_str());

    int num_cameras, num_points;

    if( write_unoptimized )
    {
        num_cameras = num_poses_;
        num_points = num_points_;
    }
    else
    {
        num_cameras = 0;
        num_points = 0;
        for (int i = 0; i < num_poses_; ++i)
            if( pose_optim_iter_[i] > 0 ) num_cameras++;

        for (int i = 0; i < num_points_; ++i)
            if( pts_optim_iter_[i] > 0 ) num_points++;
    }

    of << "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << num_cameras + num_points
       << '\n' << "property float x"
       << '\n' << "property float y"
       << '\n' << "property float z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header" << endl;

    bool write_colors = ( colors_.size() == num_points_*3 );
    if( write_unoptimized )
    {
        // Export extrinsic data (i.e. camera centers) as green points.
        double center[3];
        for (int i = 0; i < num_poses_; ++i)
        {
            const double* camera = cameraBlockPtr(i);
            cam2center (camera, center);
            of << center[0] << ' ' << center[1] << ' ' << center[2]
               << " 0 255 0" << '\n';
        }

        // Export the structure (i.e. 3D Points) as white points.
        const double* points = pointBlockPtr();
        for (int i = 0; i < num_points_; ++i)
        {
            const double* point = points + i * point_block_size_;
            for (int j = 0; j < point_block_size_; ++j)
            {
                of << point[j] << ' ';
            }
            if (write_colors )
                of << int(colors_[3*i])<<" " << int(colors_[3*i + 1])<<" "<< int(colors_[3*i + 2])<<"\n";
            else
                of << "255 255 255\n";
        }
    }
    else
    {
        // Export extrinsic data (i.e. camera centers) as green points.
        double center[3];
        for (int i = 0; i < num_poses_; ++i)
        {
            if( pose_optim_iter_[i] > 0 )
            {
                const double* camera = cameraBlockPtr(i);
                cam2center (camera, center);
                of << center[0] << ' ' << center[1] << ' ' << center[2]
                   << " 0 255 0" << '\n';
            }
        }

        // Export the structure (i.e. 3D Points) as white points.
        const double* points = pointBlockPtr();;
        for (int i = 0; i < num_points_; ++i)
        {
            if( pts_optim_iter_[i] > 0 )
            {
                const double* point = points + i * point_block_size_;
                for (int j = 0; j < point_block_size_; ++j)
                {
                    of << point[j] << ' ';
                }
                if (write_colors )
                    of << int(colors_[3*i])<<" " << int(colors_[3*i + 1])<<" "<< int(colors_[3*i + 2])<<"\n";
                else
                    of << "255 255 255\n";
            }
        }
    }
    of.close();
}

/* c_{w,cam} = R_{cam}'*[0 0 0]' - R_{cam}'*t_{cam} -> c_{w,cam} = - R_{cam}'*t_{cam} */
void BasicSfM::cam2center (const double* camera, double* center) const
{
    ConstVectorRef angle_axis_ref(camera, 3);

    Eigen::VectorXd inverse_rotation = -angle_axis_ref;
    ceres::AngleAxisRotatePoint(inverse_rotation.data(), camera + 3, center);
    VectorRef(center, 3) *= -1.0;
}

/* [0 0 0]' = R_{cam}*c_{w,cam} + t_{cam} -> t_{cam} = - R_{cam}*c_{w,cam} */
void BasicSfM::center2cam (const double* center, double* camera) const
{
    ceres::AngleAxisRotatePoint(camera, center, camera + 3);
    VectorRef(camera + 3, 3) *= -1.0;
}


bool BasicSfM::checkCheiralityConstraint (int pos_idx, int pt_idx )
{
    double *camera = cameraBlockPtr(pos_idx),
            *point = pointBlockPtr(pt_idx);

    double p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);

    // camera[5] is the z cooordinate wrt the camera at pose pose_idx
    p[2] += camera[5];
    return p[2] > 0;
}

void BasicSfM::printPose ( int idx )  const
{
    const double *cam = cameraBlockPtr(idx);
    std::cout<<"camera["<<idx<<"]"<<std::endl
             <<"{"<<std::endl
             <<"\t r_vec : ("<<cam[0]<<", "<<cam[1]<<", "<<cam[2]<<")"<<std::endl
             <<"\t t_vec : ("<<cam[3]<<", "<<cam[4]<<", "<<cam[5]<<")"<<std::endl;

    std::cout<<"}"<<std::endl;
}


void BasicSfM::printPointParams ( int idx ) const
{
    const double *pt = pointBlockPtr(idx);
    std::cout<<"point["<<idx<<"] : ("<<pt[0]<<", "<<pt[1]<<", "<<pt[2]<<")"<<std::endl;
}


void BasicSfM::solve()
{
    // Canonical camera so identity K
    cv::Mat_<double> intrinsics_matrix = cv::Mat_<double>::eye(3,3);

    // For each pose, prepare a map that reports the pairs [point index, observation index]
    vector< map<int,int> > cam_observation( num_poses_ );
    for( int i_obs = 0; i_obs < num_observations_; i_obs++ )
    {
        int i_cam = pose_index_[i_obs], i_pt = point_index_[i_obs];
        cam_observation[i_cam][i_pt] = i_obs;
    }

    // Compute a (symmetric) num_poses_ x num_poses_ matrix
    // that counts the number of correspondences between camera poses
    Eigen::MatrixXi corr = Eigen::MatrixXi::Zero(num_poses_, num_poses_);

    for( int r = 0; r < num_poses_; r++ )
    {
        for( int c = r + 1; c < num_poses_; c++ )
        {
            int nc = 0;
            for( auto const& co_iter : cam_observation[r] )
            {
                if( cam_observation[c].find( co_iter.first ) != cam_observation[c].end() )
                    nc++;
            }
            corr(r,c) = nc;
        }
    }

    Eigen::MatrixXi already_tested_pair = Eigen::MatrixXi::Zero(num_poses_, num_poses_);

    // From the correspondence matrix corr select a pair of poses that both a) maximize the number of
    // correspondences b) define an Essential Matrix as geometrical model
    bool seed_found = false;
    int ref_pose_idx = 0, new_pose_idx = 1;
    cv::Mat init_r_mat, init_r_vec, init_t;

    std::vector<cv::Point2d> points0, points1;
    cv::Mat inlier_mask_E, inlier_mask_H;

    while( !seed_found )
    {
        points0.clear();
        points1.clear();

        int max_corr = -1;
        for( int r = 0; r < num_poses_; r++ )
        {
            for (int c = r + 1; c < num_poses_; c++)
            {
                if( !already_tested_pair(r,c) && corr(r,c) > max_corr )
                {
                    max_corr = corr(r,c);
                    ref_pose_idx = r;
                    new_pose_idx = c;
                }
            }
        }

        if( max_corr < 0 )
        {
            std::cout<<"No seed pair found, exiting"<<std::endl;
            return;
        }
        already_tested_pair(ref_pose_idx,new_pose_idx) = 1;

        for (auto const &co_iter: cam_observation[ref_pose_idx])
        {
            if (cam_observation[new_pose_idx].find(co_iter.first) != cam_observation[new_pose_idx].end())
            {
                points0.emplace_back(observations_[2*co_iter.second],observations_[2*co_iter.second + 1]);
                points1.emplace_back(observations_[2*cam_observation[new_pose_idx][co_iter.first]],
                                     observations_[2*cam_observation[new_pose_idx][co_iter.first] + 1]);
            }
        }
        cv::Mat E = cv::findEssentialMat(points0, points1, intrinsics_matrix, cv::RANSAC, 0.99, 1, inlier_mask_E);
        cv::findHomography(points0, points1, inlier_mask_H, cv::LMEDS);
        int e = cv::sum(inlier_mask_E)[0];
        int h = cv::sum(inlier_mask_H)[0];

        if( e>h){
            cv::recoverPose(E, points0, points1, intrinsics_matrix, init_r_mat,init_t, inlier_mask_E);
            seed_found = true;
        }

        //////////////////////////// Code to be completed (3/5) /////////////////////////////////
        // Extract both Essential matrix E and Homograph matrix H.
        // Check that the number of inliers for the model E is higher than the number of
        // inliers for the model H (-> use inlier_mask_E and inlier_mask_H defined above <-).
        // If true, recover from E the initial rigid body transformation between ref_pose_idx
        // and new_pose_idx ( -> store it into init_r_mat and  init_t; defined above <-) and set
        // the seed_found flag to true
        // Otherwise, test a different [ref_pose_idx, new_pose_idx] pair (while( !seed_found ) loop)
        // The condition here:

        // should be replaced with the criteria described above
        /////////////////////////////////////////////////////////////////////////////////////////
    }

    // Initialize the first optimized poses, by integrating them into the registration
    // pose_optim_iter_ and pts_optim_iter_ are simple mask vectors that define which camera poses and
    // which point positions have been already registered, specifically
    // if pose_optim_iter_[pos_id] or pts_optim_iter_[id] are:
    // > 0 ---> The corresponding pose or point position has been already been estimated
    // == 0 ---> The corresponding pose or point position has not yet been estimated
    // == -1 ---> The corresponding pose or point position has been rejected due to e.g. outliers, etc...
    pose_optim_iter_[ref_pose_idx] = pose_optim_iter_[new_pose_idx] = 1;

    //Initialize the first RT wrt the reference position
    cv::Mat r_vec;
    cv::Rodrigues(init_r_mat, r_vec);
    initCamParams(new_pose_idx, r_vec, init_t );

    // Triangulate the points
    cv::Mat_<double> proj_mat0 = cv::Mat_<double>::zeros(3, 4), proj_mat1(3, 4), hpoints4D;
    proj_mat0(cv::Rect(0, 0, 3, 3)) = cv::Mat_<double>::ones(3, 3);
    proj_mat1(cv::Rect(0, 0, 3, 3)) = cv::Mat_<double>(init_r_mat);
    proj_mat1(cv::Rect(3, 0, 1, 3)) = cv::Mat_<double>(init_t);

    std::cout<<proj_mat0<<std::endl;
    cv::triangulatePoints(	proj_mat0, proj_mat1, points0, points1, hpoints4D );

    int r = 0;
    // Initialize the first optimized points
    for( auto const& co_iter : cam_observation[ref_pose_idx] )
    {
        auto &pt_idx = co_iter.first;
        if( cam_observation[new_pose_idx].find( pt_idx ) !=
            cam_observation[new_pose_idx].end() )
        {
            if( inlier_mask_E.at<unsigned char>(r) )
            {
                // Initialize the new point into the optimization
                pts_optim_iter_[pt_idx] = 1;
                double *pt = pointBlockPtr(pt_idx);

                pt[0] = hpoints4D.at<double>(0,r)/hpoints4D.at<double>(3,r);
                pt[1] = hpoints4D.at<double>(1,r)/hpoints4D.at<double>(3,r);
                pt[2] = hpoints4D.at<double>(2,r)/hpoints4D.at<double>(3,r);
            }
        }
        else
        {
            pts_optim_iter_[pt_idx] = -1;
        }
        r++;
    }

    // Start to register new poses and observations...
    for( int iter = 1; iter < num_poses_ - 1; iter++ )
    {
        // The vector n_init_pts stores the number of points already being optimized
        // that are projected in a new camera when is optimized for the first time
        std::vector<int> n_init_pts(num_poses_,0);
        int max_init_pts = -1;

        // Select the new camera (new_pose_idx) to be included in the optimization as the one that has
        // more projected points in common with the cameras already included in the optimization
        for( int i_p = 0; i_p < num_points_; i_p++ )
        {
            if( pts_optim_iter_[i_p] > 0 ) // Point already added
            {
                for( int i_c = 0; i_c < num_poses_; i_c++ )
                {
                    if( pose_optim_iter_[i_c] == 0 &&
                        cam_observation[i_c].find( i_p ) != cam_observation[i_c].end() )
                        n_init_pts[i_c]++;
                }
            }
        }
        for( int i_c = 0; i_c < num_poses_; i_c++ )
        {
            if( pose_optim_iter_[i_c] == 0 && n_init_pts[i_c] > max_init_pts )
            {
                max_init_pts = n_init_pts[i_c];
                new_pose_idx = i_c;
            }
        }

        // Now new_pose_idx is the index of the next image to be registered
        // Extract the 3D points that are projected in the new_pose_idx-th camera and that are already registered
        std::vector<cv::Point3d> scene_pts;
        std::vector<cv::Point2d> img_pts;
        for( int i_p = 0; i_p < num_points_; i_p++ )
        {
            if (pts_optim_iter_[i_p] > 0 &&
                cam_observation[new_pose_idx].find(i_p) != cam_observation[new_pose_idx].end())
            {
                double *pt = pointBlockPtr(i_p);
                scene_pts.emplace_back(pt[0], pt[1], pt[3]);
                img_pts.emplace_back(observations_[cam_observation[new_pose_idx][i_p] * 2],
                                     observations_[cam_observation[new_pose_idx][i_p] * 2 + 1]);
            }
        }

        if( scene_pts.size() <= 3 )
        {
            std::cout<<"No other positions can be optimized, exiting"<<std::endl;
            return;
        }

        // Estimate an initial R,t by using PnP + RANSAC
        cv::solvePnPRansac(scene_pts, img_pts, intrinsics_matrix, cv::Mat(), init_r_vec, init_t);
        initCamParams(new_pose_idx, init_r_vec, init_t);
        // ... and add to the pool of optimized camera positions
        pose_optim_iter_[new_pose_idx] = 1;

        // Extract the new points that, thanks to the new camera, are going to be optimized
        int n_new_pts = 0;
        for( int cam_idx = 0; cam_idx < num_poses_; cam_idx++ )
        {
            if( pose_optim_iter_[cam_idx] > 0 )
            {
                for( auto const& co_iter : cam_observation[cam_idx] )
                {
                    auto &pt_idx = co_iter.first;
                    if( pts_optim_iter_[pt_idx] == 0 &&
                        cam_observation[new_pose_idx].find( pt_idx ) != cam_observation[new_pose_idx].end() )
                    {
                        n_new_pts++;
                        pts_optim_iter_[pt_idx] = 1;
                        // Point are simply initialized with a distance of 1 "meter" from one camera that frame such point
                        initPointParams(pt_idx, cam_idx, observations_.data() + 2 * co_iter.second, 1.0 );
                    }
                }
            }
        }

        cout<<"ADDED "<<n_new_pts<<" new points"<<endl;

        cout<<"Using "<<iter + 2<<" over "<<num_poses_<<" cameras"<<endl;
        for( int i = 0; i < int( pose_optim_iter_.size()); i++ )
            cout<<int( pose_optim_iter_[i])<<" ";
        cout<<endl;

        bundleAdjustmentIter( new_pose_idx );

        const int  max_dist = 10;
        double *pts = parameters_.data() + num_poses_ * camera_block_size_;
        for( int i = 0; i < num_points_; i++ )
        {
            if( pts_optim_iter_[i] > 0 &&
                ( fabs(pts[i*point_block_size_]) > max_dist ||
                  fabs(pts[i*point_block_size_ + 1]) > max_dist ||
                  fabs(pts[i*point_block_size_ + 2]) > max_dist ) )
            {
                pts_optim_iter_[i] = -1;
            }
        }
    }


}

void BasicSfM::initCamParams(int new_pose_idx, cv::Mat r_vec, cv::Mat t_vec )
{
    double *camera = cameraBlockPtr(new_pose_idx);

    cv::Mat_<double> r_vec_d(r_vec), t_vec_d(t_vec);
    for( int r = 0; r < 3; r++ )
    {
        camera[r] = r_vec_d(r,0);
        camera[r+3] = t_vec_d(r,0);
    }
}

void BasicSfM::initPointParams(int pt_idx, int pos_idx, const double img_p[2], double depth)
{
    double *camera = cameraBlockPtr(pos_idx);

    Eigen::Vector3d rel_pos, pos;
    rel_pos(0) = img_p[0]*depth;
    rel_pos(1) = img_p[1]*depth;
    rel_pos(2) = depth;

    // cam_r_vec and cam_t_vec represent the transformation that maps point from the cam_idx-th camera
    // frame to the world frame
    Eigen::Vector3d cam_r_vec, cam_t_vec;
    cam_r_vec << -camera[0], -camera[1], -camera[2];

    ceres::AngleAxisRotatePoint(cam_r_vec.data(), camera + 3, cam_t_vec.data());

    // Represent the point rel_pos in the world frame
    ceres::AngleAxisRotatePoint(cam_r_vec.data(), rel_pos.data(), pos.data());
    pos -= cam_t_vec;

    double *pt = pointBlockPtr(pt_idx);

    pt[0] = pos(0);
    pt[1] = pos(1);
    pt[2] = pos(2);
}

void BasicSfM::bundleAdjustmentIter( int new_cam_idx )
{
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 4;
    options.max_num_iterations = 200;

    std::vector<double> bck_parameters;

    bool keep_optimize = true;

    // Global optimization
    while( keep_optimize )
    {
        bck_parameters = parameters_;

        ceres::Problem problem;
        // For each observation....
        for( int i_obs = 0; i_obs < num_observations_; i_obs++ )
        {
            //.. check if this observation has bem already registered (bot checking camera pose and point pose)
            if( pose_optim_iter_[pose_index_[i_obs]] > 0 && pts_optim_iter_[point_index_[i_obs]] > 0 )
            {

                auto b = parameters_.data()[i_obs] + camera_block_size_ * i_obs;
                auto e = b + camera_block_size_;
                std::cout<<"b)"<<b<<std::endl;
                std::cout<<"e"<<e<<std::endl;
                std::vector<double> camera;
                for (auto i=b; i<e; i++) {
                    camera.push_back(parameters_[i]);
                    std::cout<<"camera)"<<camera[i]<<std::endl;
                }
                std::cout<<"*******************"<<std::endl;
                std::cout<<"pointBlockPtr (point_index_[i_obs]))"<<pointBlockPtr (point_index_[i_obs])<<std::endl;

                ceres::CostFunction* cost_function =
                        ReprojectionError::Create(
                                observations_[2 * i_obs + 0],
                                observations_[2 * i_obs + 1]);

                problem.AddResidualBlock(cost_function,
                                         new ceres::CauchyLoss(2*max_reproj_err_),
                                         parameters_.data()+ camera_block_size_ * i_obs);
                //////////////////////////// Code to be completed (4/5) /////////////////////////////////
                //.. in case, add a residual block inside the Ceres solver problem.
                // You should define a suitable functor (i.e., see the ReprojectionError struct at the
                // beginning of this file)
                // You may try a Cauchy loss function with parameters, say 2*max_reproj_err_
                // Remember that the parameter blocks are stored starting from the
                // parameters_.data() double* pointer.
                // The camera position blocks have size (camera_block_size_) of 6 elements,
                // while the point position blocks have size (point_block_size_) of 3 elements.
                //////////////////////////////////////////////////////////////////////////////////
            }
        }

        ceres::Solver::Summary summary;
        Solve(options, &problem, &summary);
//    if( verbosity_level_ > 2)
//      std::cout << summary.FullReport() << "\n";



        bool cheirality_violation = false;
        // TODO Optmize here
        int dbg_n_violations = 0;
        for( int i_obs = 0; i_obs < num_observations_; i_obs++ )
        {
            if( pose_optim_iter_[pose_index_[i_obs]] > 0 &&
                pts_optim_iter_[point_index_[i_obs]] > 0 &&
                !checkCheiralityConstraint(pose_index_[i_obs], point_index_[i_obs]))
            {
                // Remove point from the optimization
                pts_optim_iter_[point_index_[i_obs]] = -1;
                cheirality_violation = true;
                dbg_n_violations++;
            }
        }

        if( cheirality_violation )
        {
            std::cout<<"****************** OPTIM CHEIRALITY VIOLATION for "<<dbg_n_violations<<" points : redoing optim!!"<<std::endl;
            parameters_ = bck_parameters;
//      memcpy(parameters_, bck_parameters_, num_parameters_*sizeof(double));
        }
        else if ( rejectOuliers() > max_outliers_ )
        {
            std::cout<<"****************** OPTIM TOO MANY OUTLIERS: redoing optim!!"<<std::endl;
            parameters_ = bck_parameters;
        }
        else
            keep_optimize = false;
    }

    for( auto &c_count : pose_optim_iter_ )
        if( c_count > 0 ) c_count++;

    for( auto &p_count : pts_optim_iter_ )
        if( p_count > 0 ) p_count++;


    if(new_cam_idx >= 0)
        printPose ( new_cam_idx );
}

int BasicSfM::rejectOuliers()
{
    int num_ouliers = 0;
    for( int i_obs = 0; i_obs < num_observations_; i_obs++ )
    {
        if( pose_optim_iter_[pose_index_[i_obs]] > 0 && pts_optim_iter_[point_index_[i_obs]] > 0 )
        {
            double *camera = cameraBlockPtr (pose_index_[i_obs]),
                    *point = pointBlockPtr (point_index_[i_obs]),
                    *observation = observations_.data() + (i_obs * 2);

            double p[3];
            ceres::AngleAxisRotatePoint(camera, point, p);

            // camera[3,4,5] are the translation.
            p[0] += camera[3];
            p[1] += camera[4];
            p[2] += camera[5];

            double predicted_x = p[0] / p[2];
            double predicted_y = p[1] / p[2];

            if ( fabs(predicted_x - observation[0]) > max_reproj_err_ ||
                 fabs(predicted_y - observation[1]) > max_reproj_err_ )
            {
                pts_optim_iter_[point_index_[i_obs]] = -1;
                num_ouliers ++;
            }
        }
    }
    std::cout<<"--------------> REJECTED "<<num_ouliers <<" OUTLIERS\n";
    return num_ouliers;
}