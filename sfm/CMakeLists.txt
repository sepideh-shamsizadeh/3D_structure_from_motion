project (sfm)

cmake_minimum_required (VERSION 2.8)

set(CMAKE_BUILD_TYPE "Release")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=gnu++11 -g")

find_package( Boost COMPONENTS filesystem REQUIRED )
find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
find_package( Ceres REQUIRED)

#Add here your source files
set(3DP_SFM_SRCS src/io_utils.cpp src/features_matcher.cpp src/basic_sfm.cpp)

add_library(${PROJECT_NAME} ${3DP_SFM_SRCS})
target_include_directories( ${PROJECT_NAME} PUBLIC
                            src
                            ${OpenCV_INCLUDE_DIRS}
                            ${EIGEN3_INCLUDE_DIR}
                            ${CERES_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
                      ${Boost_LIBRARIES}
                      ${CERES_LIBRARIES}
                      ${OpenCV_LIBS})

add_executable(matcher src/matcher_app.cpp)
target_link_libraries(matcher ${PROJECT_NAME})
set_target_properties(matcher PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(basic_sfm src/sfm_app.cpp)
target_link_libraries(basic_sfm ${PROJECT_NAME})
set_target_properties(basic_sfm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
