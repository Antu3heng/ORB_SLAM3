cmake_minimum_required(VERSION 3.0.2)
project(orb_slam3)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3  -Wl,--copy-dt-needed-entries")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3  -Wl,--copy-dt-needed-entries")

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  tf
)

find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 REQUIRED)

catkin_package()

include_directories(
${catkin_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
${OpenCV_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/../../../
${PROJECT_SOURCE_DIR}/../../../include
${PROJECT_SOURCE_DIR}/../../../include/CameraModels
)

set(LIBS 
${OpenCV_LIBS} 
${EIGEN3_LIBS}
${Boost_LIBRARIES}
${PROJECT_SOURCE_DIR}/../../../lib/libORB_SLAM3.so
)

add_executable(orb_rgbd_inertial src/orb_rgbd_inertial.cpp)
target_link_libraries(orb_rgbd_inertial
  ${catkin_LIBRARIES}
  ${LIBS}
)

add_executable(ros_stereo_inertial src/ros_stereo_inertial.cc)
target_link_libraries(ros_stereo_inertial
  ${catkin_LIBRARIES}
  ${LIBS}
)

add_executable(ros_stereo src/ros_stereo.cc)
target_link_libraries(ros_stereo
  ${catkin_LIBRARIES}
  ${LIBS}
)

add_executable(ros_rgbd_inertial src/ros_rgbd_inertial.cc)
target_link_libraries(ros_rgbd_inertial
  ${catkin_LIBRARIES}
  ${LIBS}
)
