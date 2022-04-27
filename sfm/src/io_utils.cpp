#include "io_utils.h"

#include <fstream>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

using namespace std;
using namespace boost;
using namespace boost::filesystem;

bool loadCameraParams( const std::string &file_name, cv::Size &image_size,
                       cv::Mat &camera_matrix, cv::Mat &dist_coeffs )
{
  cv::FileStorage fs(file_name, cv::FileStorage::READ);

  if( !fs.isOpened() )
    return false;

  fs["width"]>>image_size.width;
  fs["height"]>>image_size.height;

  fs["K"]>>camera_matrix;
  fs["D"]>>dist_coeffs;

  fs.release();

  return true;
}

bool readFileNamesFromFolder ( const string& input_folder_name, vector< string >& names )
{
  names.clear();
  if ( !input_folder_name.empty() )
  {
    path p ( input_folder_name );
    for ( auto& entry : make_iterator_range ( directory_iterator ( p ), {} ) )
      names.push_back ( entry.path().string() );
    std::sort ( names.begin(), names.end() );
    return true;
  }
  else
  {
    return false;
  }
}
