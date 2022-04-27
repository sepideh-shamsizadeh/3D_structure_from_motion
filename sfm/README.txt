Prerequisites (in debian-based distro):

sudo apt install build-essential cmake libboost-filesystem-dev libopencv-dev libomp-dev libceres-dev libyaml-cpp-dev libgtest-dev libeigen3-dev

Build and run the executable:

mkdir build
cd build
cmake ..
make

./basic_sfm <input data file> <output ply file>
./matcher <calibration parameters filename> <images folder filename><output data file> [focal length scale]


