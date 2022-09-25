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

#include <opencv2/core/core.hpp>

#include "../../../include/System.h"
#include "../include/ImuTypes.h"

using namespace std;

class ImuGrabber
{
public:
    ImuGrabber(){};
    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System *pSLAM, ImuGrabber *pImuGb, const bool bClahe) : mpSLAM(pSLAM), mpImuGb(pImuGb), mbClahe(bClahe) {}

    void GrabImageRgb(const sensor_msgs::ImageConstPtr &msg);
    void GrabImageDepth(const sensor_msgs::ImageConstPtr &msg);
    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);
    void SyncWithImu();

    queue<sensor_msgs::ImageConstPtr> rgbBuf, depthBuf;
    std::mutex mBufMutexRgb, mBufMutexDepth;

    ORB_SLAM3::System *mpSLAM;
    ImuGrabber *mpImuGb;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "orb_rgbd_inertial");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    bool bEqual = false;
    if (argc < 3 || argc > 4)
    {
        cerr << endl
             << "Usage: rosrun orb_slam3 orb_rgbd_inertial path_to_vocabulary path_to_settings [do_equalize]" << endl;
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

    ImuGrabber imugb;
    ImageGrabber igb(&SLAM, &imugb, bEqual);

    // Maximum delay, 5 seconds
    ros::Subscriber sub_imu = n.subscribe("/d400/imu", 1000, &ImuGrabber::GrabImu, &imugb);
    ros::Subscriber sub_img_rgb = n.subscribe("/d400/color/image_raw", 100, &ImageGrabber::GrabImageRgb, &igb);
    ros::Subscriber sub_img_depth = n.subscribe("/d400/aligned_depth_to_color/image_raw", 100, &ImageGrabber::GrabImageDepth, &igb);

    std::thread sync_thread(&ImageGrabber::SyncWithImu, &igb);

    ros::spin();

    return 0;
}

void ImageGrabber::GrabImageRgb(const sensor_msgs::ImageConstPtr &img_msg)
{
    mBufMutexRgb.lock();
    if (!rgbBuf.empty())
        rgbBuf.pop();
    rgbBuf.push(img_msg);
    mBufMutexRgb.unlock();
}

void ImageGrabber::GrabImageDepth(const sensor_msgs::ImageConstPtr &img_msg)
{
    mBufMutexDepth.lock();
    if (!depthBuf.empty())
        depthBuf.pop();
    depthBuf.push(img_msg);
    mBufMutexDepth.unlock();
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(img_msg);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    return cv_ptr->image.clone();

    // if (cv_ptr->image.type() == 0)
    // {
    //     return cv_ptr->image.clone();
    // }
    // else
    // {
    //     std::cout << "Error type" << std::endl;
    //     return cv_ptr->image.clone();
    // }
}

void ImageGrabber::SyncWithImu()
{
    const double maxTimeDiff = 0.01;
    while (1)
    {
        cv::Mat imRgb, imDepth;
        double tImRgb = 0, tImDepth = 0;
        if (!rgbBuf.empty() && !depthBuf.empty() && !mpImuGb->imuBuf.empty())
        {
            tImRgb = rgbBuf.front()->header.stamp.toSec();
            tImDepth = depthBuf.front()->header.stamp.toSec();

            this->mBufMutexDepth.lock();
            while ((tImRgb - tImDepth) > maxTimeDiff && depthBuf.size() > 1)
            {
                depthBuf.pop();
                tImDepth = depthBuf.front()->header.stamp.toSec();
            }
            this->mBufMutexDepth.unlock();

            this->mBufMutexRgb.lock();
            while ((tImDepth - tImRgb) > maxTimeDiff && rgbBuf.size() > 1)
            {
                rgbBuf.pop();
                tImRgb = rgbBuf.front()->header.stamp.toSec();
            }
            this->mBufMutexRgb.unlock();

            if ((tImRgb - tImDepth) > maxTimeDiff || (tImDepth - tImRgb) > maxTimeDiff)
            {
                // std::cout << "big time difference" << std::endl;
                continue;
            }
            if (tImRgb > mpImuGb->imuBuf.back()->header.stamp.toSec())
                continue;

            this->mBufMutexRgb.lock();
            imRgb = GetImage(rgbBuf.front());
            rgbBuf.pop();
            this->mBufMutexRgb.unlock();

            this->mBufMutexDepth.lock();
            imDepth = GetImage(depthBuf.front());
            depthBuf.pop();
            this->mBufMutexDepth.unlock();

            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            mpImuGb->mBufMutex.lock();
            if (!mpImuGb->imuBuf.empty())
            {
                // Load imu measurements from buffer
                vImuMeas.clear();
                while (!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec() <= tImRgb)
                {
                    double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
                    cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x, mpImuGb->imuBuf.front()->linear_acceleration.y, mpImuGb->imuBuf.front()->linear_acceleration.z);
                    cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x, mpImuGb->imuBuf.front()->angular_velocity.y, mpImuGb->imuBuf.front()->angular_velocity.z);
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                    mpImuGb->imuBuf.pop();
                }
            }
            mpImuGb->mBufMutex.unlock();
            if (mbClahe)
            {
                mClahe->apply(imRgb, imRgb);
                mClahe->apply(imDepth, imDepth);
            }

            mpSLAM->TrackRGBD(imRgb, imDepth, tImRgb, vImuMeas);

            std::chrono::milliseconds tSleep(1);
            std::this_thread::sleep_for(tSleep);
        }
    }
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
    mBufMutex.lock();
    imuBuf.push(imu_msg);
    mBufMutex.unlock();
    return;
}