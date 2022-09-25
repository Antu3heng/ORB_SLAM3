/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/core/core.hpp>

#include "../../../include/System.h"
#include "../include/ImuTypes.h"

using namespace std;

class ImuGrabber
{
public:
    ImuGrabber(ros::NodeHandle nh, ORB_SLAM3::System *pSLAM) : n_(nh), mpSLAM(pSLAM)
    {
        odometry_pub = n_.advertise<nav_msgs::Odometry>("/orb_slam3/odom", 1000);
    }

    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    ORB_SLAM3::System *mpSLAM;
    ros::NodeHandle n_;
    ros::Publisher odometry_pub;
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System *pSLAM) : mpSLAM(pSLAM)
    {
        sync_processing = std::thread(&ImageGrabber::sync_process, this);
    }

    void GrabStereo(const sensor_msgs::ImageConstPtr &msgLeft, const sensor_msgs::ImageConstPtr &msgRight);

    void sync_process();

    ORB_SLAM3::System *mpSLAM;
    std::thread sync_processing;
    bool do_rectify;
    cv::Mat M1l, M2l, M1r, M2r;
    std::mutex m_buf;
    queue<sensor_msgs::ImageConstPtr> img0_buf;
    queue<sensor_msgs::ImageConstPtr> img1_buf;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ros_stereo_inertial");
    ros::start();

    if (argc != 4)
    {
        cerr << endl
             << "Usage: rosrun orb_slam3 ros_stereo_inertial path_to_vocabulary path_to_settings do_rectify" << endl;
        ros::shutdown();
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, true);

    ImageGrabber igb(&SLAM);

    stringstream ss(argv[3]);
    ss >> boolalpha >> igb.do_rectify;

    if (igb.do_rectify)
    {
        // Load settings related to stereo calibration
        cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
        if (!fsSettings.isOpened())
        {
            cerr << "ERROR: Wrong path to settings" << endl;
            return -1;
        }

        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
        fsSettings["LEFT.K"] >> K_l;
        fsSettings["RIGHT.K"] >> K_r;

        fsSettings["LEFT.P"] >> P_l;
        fsSettings["RIGHT.P"] >> P_r;

        fsSettings["LEFT.R"] >> R_l;
        fsSettings["RIGHT.R"] >> R_r;

        fsSettings["LEFT.D"] >> D_l;
        fsSettings["RIGHT.D"] >> D_r;

        int rows_l = fsSettings["LEFT.height"];
        int cols_l = fsSettings["LEFT.width"];
        int rows_r = fsSettings["RIGHT.height"];
        int cols_r = fsSettings["RIGHT.width"];

        if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
            rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0)
        {
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            return -1;
        }

        cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, igb.M1l, igb.M2l);
        cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F, igb.M1r, igb.M2r);
    }

    ros::NodeHandle nh;

    ImuGrabber imugb(nh, &SLAM);

    ros::Subscriber imu_sub = nh.subscribe("/imu0", 1000, &ImuGrabber::GrabImu, &imugb);
    message_filters::Subscriber<sensor_msgs::Image> left_sub(nh, "/cam0/image_raw", 10);
    message_filters::Subscriber<sensor_msgs::Image> right_sub(nh, "/cam1/image_raw", 10);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), left_sub, right_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabStereo, &igb, _1, _2));

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_TUM_Format.txt");
    SLAM.SaveTrajectoryTUM("FrameTrajectory_TUM_Format.txt");
    SLAM.SaveTrajectoryKITTI("FrameTrajectory_KITTI_Format.txt");

    ros::shutdown();

    return 0;
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    cv::Point3f acc(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
    cv::Point3f gyr(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
    ORB_SLAM3::IMU::Point imuData(acc, gyr, t);
    auto pvq = mpSLAM->TrackIMU(imuData);

    if (pvq.is_valid)
    {
        nav_msgs::Odometry odometry;
        odometry.header.stamp = ros::Time(t);
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = pvq.pos.x();
        odometry.pose.pose.position.y = pvq.pos.y();
        odometry.pose.pose.position.z = pvq.pos.z();
        odometry.pose.pose.orientation.x = pvq.q.x();
        odometry.pose.pose.orientation.y = pvq.q.y();
        odometry.pose.pose.orientation.z = pvq.q.z();
        odometry.pose.pose.orientation.w = pvq.q.w();
        odometry.twist.twist.linear.x = pvq.vel.x();
        odometry.twist.twist.linear.y = pvq.vel.y();
        odometry.twist.twist.linear.z = pvq.vel.z();
        odometry_pub.publish(odometry);
    }
}

void ImageGrabber::GrabStereo(const sensor_msgs::ImageConstPtr &msgLeft, const sensor_msgs::ImageConstPtr &msgRight)
{
    unique_lock<std::mutex> lock(m_buf);
    img0_buf.push(msgLeft);
    img1_buf.push(msgRight);
}

void ImageGrabber::sync_process()
{
    while (true)
    {
        sensor_msgs::ImageConstPtr msgLeft, msgRight;
        int flag = false;
        {
            unique_lock<std::mutex> lock(m_buf);
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                msgLeft = img0_buf.front();
                img0_buf.pop();
                msgRight = img1_buf.front();
                img1_buf.pop();
                flag = true;
            }
        }

        if (flag)
        {
            // Copy the ros image message to cv::Mat.
            cv_bridge::CvImageConstPtr cv_ptrLeft;
            try
            {
                cv_ptrLeft = cv_bridge::toCvShare(msgLeft);
            }
            catch (cv_bridge::Exception &e)
            {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }

            cv_bridge::CvImageConstPtr cv_ptrRight;
            try
            {
                cv_ptrRight = cv_bridge::toCvShare(msgRight);
            }
            catch (cv_bridge::Exception &e)
            {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }

            if (do_rectify)
            {
                cv::Mat imLeft, imRight;
                cv::remap(cv_ptrLeft->image, imLeft, M1l, M2l, cv::INTER_LINEAR);
                cv::remap(cv_ptrRight->image, imRight, M1r, M2r, cv::INTER_LINEAR);
                mpSLAM->TrackStereo(imLeft, imRight, cv_ptrLeft->header.stamp.toSec());
            }
            else
            {
                mpSLAM->TrackStereo(cv_ptrLeft->image, cv_ptrRight->image, cv_ptrLeft->header.stamp.toSec());
            }
        }
    }
}