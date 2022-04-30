
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <string>

#include <iostream>

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{5,7};

int main()
{
    // Creating vector to store vectors of 3D points for each checkerboard image
    std::vector<std::vector<cv::Point3f> > objpoints;

    // Creating vector to store vectors of 2D points for each checkerboard image
    std::vector<std::vector<cv::Point2f> > imgpoints;

    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for(int i{0}; i<CHECKERBOARD[1]; i++)
    {
        for(int j{0}; j<CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(j*28.5,i*28.5,0));
    }


    // Extracting path of individual image stored in a given directory
    std::vector<cv::String> images;
    // Path of the folder containing checkerboard images
    std::string path = "../images/*.png";

    cv::glob(path, images);

    cv::Mat frame, gray;
    // vector to store the pixel coordinates of detected checker board corners
    std::vector<cv::Point2f> corner_pts;
    bool success;

    // Looping over all the images in the directory
    for(int i{0}; i<images.size(); i++)
    {
        frame = cv::imread(images[i]);
        cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);

        // Finding checker board corners
        // If desired number of corners are found in the image then success = true
        success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
                                                                                                          + cv::CALIB_CB_FAST_CHECK);

        /*
         * If desired number of corner are detected,
         * we refine the pixel coordinates and display
         * them on the images of checker board
        */
        if(success)
        {
            cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,30, 0.001);

            // refining pixel coordinates for given 2d points.
            cv::cornerSubPix(gray,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);

            // Displaying the detected corner points on the checker board
            cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }

        cv::imshow("Image",frame);
        cv::waitKey(0);
    }

    cv::destroyAllWindows();

    cv::Mat cameraMatrix,distCoeffs,R,T;

    cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T);

    std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
    std::cout << "distCoeffs : " << distCoeffs << std::endl;
    std::cout << "Rotation vector : " << R << std::endl;
    std::cout << "Translation vector : " << T << std::endl;

    cv::Mat image_result = cv::imread("../test1.png");
    cv::imshow("original undistorted", image_result);
    cv::Mat imageUndistorted, new_camera_matrix;
    cv::Size imageSize(cv::Size(image_result.cols,image_result.rows));
    new_camera_matrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0);
    cv::undistort( image_result, imageUndistorted, new_camera_matrix, distCoeffs, new_camera_matrix );

    cv::imshow("original image", imageUndistorted);
    cv::waitKey(0);
    std::string filename = "../../configs/parameters.yml";
    { //write
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        // or:
        // FileStorage fs;
        // fs.open(filename, FileStorage::WRITE);
        fs << "cameraMatrix" << cameraMatrix;
        fs << "distCoeffs" << distCoeffs ;                              // text - string sequence
        fs << "Rotation vector" << R;
        fs << "Translation vector" << T ;                                           // close sequence
        // your own data structures
        fs.release();                                       // explicit close
        std::cout << "Write Done." << std::endl;
    }
    return 0;
}