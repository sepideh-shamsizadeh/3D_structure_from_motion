#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "basic_sfm.h"

int main(int argc, char **argv)
{
  if( argc < 3 )
  {
    std::cout<<"Usage : "<<argv[0]<<" <input data file> <output ply file>"<<std::endl;
    return 0;
  }
  std::string input_file(argv[1]);

  BasicSfM sfm;
  sfm.readFromFile(input_file, false, true );
  sfm.solve();
  sfm.writeToPLYFile(argv[2]);

  return 0;
}
