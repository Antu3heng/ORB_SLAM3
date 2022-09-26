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
    ImageGrabber(ORB_SLAM3::System *pSLAM, const bool bClahe) : mpSLAM(pSLAM), mbClahe(bClahe)
    {
        sync_processing = std::thread(&ImageGrabber::sync_process, this);
    }

    void GrabRGBD(const sensor_msgs::ImageConstPtr &msgRgb, const sensor_msgs::ImageConstPtr &msgDepth);

    void sync_process();

    ORB_SLAM3::System *mpSLAM;
    std::thread sync_processing;
    std::mutex m_buf;

    queue<sensor_msgs::ImageConstPtr> rgb_buf;
    queue<sensor_msgs::ImageConstPtr> depth_buf;

    //Improves image contrast
    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
};



int main(int argc, char **argv)
{
    ros::init(argc, argv, "ros_rgbd_inertial");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    bool bEqual = false;
    bool flag;
    if (argc < 3 || argc > 4)
    {
        cerr << endl
             << "Usage: rosrun orb_slam3 ros_rgbd_inertial path_to_vocabulary path_to_settings [do_equalize]" << endl;
        ros::shutdown();
        return 1;
    }

    if (argc == 4)
    {
        std::string sbEqual(argv[3]);
        if (sbEqual == "true")
            bEqual = true;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_RGBD, true);

    ImuGrabber imugb(n, &SLAM);
    ImageGrabber igb(&SLAM, bEqual);

    ros::Subscriber imu_sub = n.subscribe("/airsim_node/drone_1/imu/imu", 1000, &ImuGrabber::GrabImu, &imugb);
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(n, "/airsim_node/drone_1/front_center/Scene", 10);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(n, "/airsim_node/drone_1/front_center/DepthPlanar", 10);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub, depth_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD, &igb, _1, _2));
    
    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_TUM_Format.txt");
    SLAM.SaveTrajectoryTUM("FrameTrajectory_TUM_Format.txt");
    SLAM.SaveTrajectoryKITTI("FrameTrajectory_KITTI_Format.txt");
    
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

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr &msgRgb, const sensor_msgs::ImageConstPtr &msgDepth)
{
    unique_lock<std::mutex> lock(m_buf);
    rgb_buf.push(msgRgb);
    depth_buf.push(msgDepth);
}

void ImageGrabber::sync_process()
{
    while (true)
    {
        sensor_msgs::ImageConstPtr msgRgb, msgDepth;
        int flag = false;
        {
            unique_lock<std::mutex> lock(m_buf);
            if (!rgb_buf.empty() && !depth_buf.empty())
            {
                msgRgb = rgb_buf.front();
                rgb_buf.pop();
                msgDepth = depth_buf.front();
                depth_buf.pop();
                flag = true;
            }
        }

        if (flag)
        {
            // Copy the ros image message to cv::Mat.
            cv_bridge::CvImageConstPtr cv_ptrRgb;
            try
            {
                cv_ptrRgb = cv_bridge::toCvShare(msgRgb);
            }
            catch (cv_bridge::Exception &e)
            {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }

            cv_bridge::CvImageConstPtr cv_ptrDepth;
            try
            {
                cv_ptrDepth = cv_bridge::toCvShare(msgDepth);
            }
            catch (cv_bridge::Exception &e)
            {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }

            cv::Mat imRgb;
            cv::resize(cv_ptrRgb->image, imRgb, cv::Size(320,240), 0.5, 0.5, CV_INTER_AREA);

            mpSLAM->TrackRGBD(imRgb, cv_ptrDepth->image, cv_ptrRgb->header.stamp.toSec());
        }
    }
}