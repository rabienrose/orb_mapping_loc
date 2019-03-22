/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/



#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <rosbag/bag.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CompressedImage.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>

namespace ORB_SLAM2
{

    
std::vector<std::string> split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}
    
System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer, bool bReuse, string mapFilePath):
               mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), mbReset(false),mbActivateLocalizationMode(false),
                mbDeactivateLocalizationMode(false)
{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }


    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    mpVocabulary = new ORBVocabulary();
    bool bVocLoad= mpVocabulary->loadFromBinaryFile(strVocFile);
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    //Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);
    
    if (!bReuse)
    {
        mpMap = new Map();
    }
    else
    {
        LoadMap(mapFilePath.c_str());
        std::cout<<"succ load map!"<<std::endl;
        //mpKeyFrameDatabase->set_vocab(mpVocabulary);

        vector<ORB_SLAM2::KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
        for (vector<ORB_SLAM2::KeyFrame*>::iterator it = vpKFs.begin(); it != vpKFs.end(); ++it)
        {
            (*it)->SetKeyFrameDatabase(mpKeyFrameDatabase);
            (*it)->SetORBvocabulary(mpVocabulary);
            (*it)->SetMap(mpMap);
            (*it)->ComputeBoW();
            mpKeyFrameDatabase->add(*it);
            (*it)->SetMapPoints(mpMap->GetAllMapPoints());
            (*it)->SetSpanningTree(vpKFs);
            (*it)->SetGridParams(vpKFs);

            // Reconstruct map points Observation
        }

        vector<ORB_SLAM2::MapPoint*> vpMPs = mpMap->GetAllMapPoints();
        for (vector<ORB_SLAM2::MapPoint*>::iterator mit = vpMPs.begin(); mit != vpMPs.end(); ++mit)
        {
            (*mit)->SetMap(mpMap);
            (*mit)->SetObservations(vpKFs);
        }

        for (vector<ORB_SLAM2::KeyFrame*>::iterator it = vpKFs.begin(); it != vpKFs.end(); ++it)
        {
            (*it)->UpdateConnections();
        }
    }

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor, bReuse);

    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);

    //Initialize the Loop Closing thread and launch
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);

    //Initialize the Viewer thread and launch
    if(bUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
    }

    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
{
    if(mSensor!=STEREO)
    {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
        exit(-1);
    }   

    // Check mode change
    {
        //unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    //unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft,imRight,timestamp);

    //unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp)
{
    if(mSensor!=RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    // Check mode change
    {
        //unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    //unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageRGBD(im,depthmap,timestamp);

    //unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp, std::string file_name)
{
    if(mSensor!=MONOCULAR)
    {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
        exit(-1);
    }

    // Check mode change
    {
        //unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    //unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }
    cv::Mat Tcw = mpTracker->GrabImageMonocular(im,timestamp, file_name);

    //unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}

void System::ActivateLocalizationMode()
{
    //unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    //unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    //unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        usleep(5000);
    }

    if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

void System::LoadMap(const string &filename){
    {
        std::ifstream is(filename);

       
        boost::archive::binary_iarchive ia(is, boost::archive::no_header);
        //ia >> mpKeyFrameDatabase;
        ia >> mpMap;
       
    }

    cout << endl << filename <<" : Map Loaded!" << endl;
//     std::ifstream infile(filename);
//     std::string line;
//     std::getline(infile, line);
//     int kf_count=std::stoi (line);
//     std::getline(infile, line);
//     int mp_count=std::stoi (line);
//     for(size_t i=0; i<kf_count; i++){
//         std::getline(infile, line);
//         int line_count=std::stoi (line);
//         std::vector<std::string> kf_str;
//         for(int i=0;i<line_count;i++){
//             std::getline(infile, line);
//             kf_str.push_back(line);
//         }
//         KeyFrame* pKF= new KeyFrame(kf_str, mpMap,mpKeyFrameDatabase);
//     }
//     std::cout<<"got me"<<std::endl;
//     for(size_t i=0; i<mp_count; i++){
//         
//     }
}

void System::SaveMap(const string &filename)
{
    std::ofstream os(filename);
    {
        ::boost::archive::binary_oarchive oa(os, ::boost::archive::no_header);
        //oa << mpKeyFrameDatabase;
        oa << mpMap;
    }
    cout << endl << "Map saved to " << filename << endl;
//     cout << endl << "Saving map to " << filename << " ..." << endl;
// 
//     vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
//     vector<MapPoint*> vpMPs= mpMap->GetAllMapPoints();
//     int goodkf_count=0;
//     int goodmp_count=0;
//     for(size_t i=0; i<vpKFs.size(); i++){
//         if(vpKFs[i]->isBad())
//             continue;
//         goodkf_count++;        
//     }
//     for(size_t i=0; i<vpMPs.size(); i++)
//     {
//         if(vpMPs[i]->isBad())
//             continue;
//         goodmp_count++;   
//     }
//     sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);
// 
//     ofstream f;
//     f.open(filename.c_str());
//     f << fixed;
//     f<<goodkf_count<<std::endl;
//     f<<goodmp_count<<std::endl;
// 
//     for(size_t i=0; i<vpKFs.size(); i++)
//     {
//         KeyFrame* pKF = vpKFs[i];
//         if(pKF->isBad())
//             continue;
//         std::string data_str=pKF->getKFDataStr();
//         f<<data_str;
//     }
//     for(size_t i=0; i<vpMPs.size(); i++)
//     {
//         MapPoint* pMP = vpMPs[i];
//         if(pMP->isBad())
//             continue;
//         std::string data_str=pMP->getMPDataStr();
//         f<<data_str;
//     }
// 
//     f.close();
//     cout << endl << "map saved!" << endl;
}

void convert2MsgCloud(std::vector<cv::Mat> posi, sensor_msgs::PointCloud2& cloud_oud){
    sensor_msgs::PointCloud2& cloud=cloud_oud;
    sensor_msgs::PointField field;
    field.name='x';
    field.offset=0;
    field.datatype=sensor_msgs::PointField::FLOAT32;
    field.count=1;
    cloud.fields.push_back(field);
    field.name='y';
    field.offset=4;
    cloud.fields.push_back(field);
    field.name='z';
    field.offset=8;
    cloud.fields.push_back(field);
    cloud.height=1;
    cloud.width=posi.size();
    cloud.point_step=12;
    cloud.row_step=cloud.width;
    cloud.is_dense=true;
    cloud.is_bigendian=false;
    cloud.header.frame_id="map";
    for(int i=0; i<posi.size(); i++){
        cv::Mat posi_mat=posi[i];
        float x=posi_mat.at<float>(0);
        float y=posi_mat.at<float>(1);
        float z=posi_mat.at<float>(2);
        unsigned char const * pcx = reinterpret_cast<unsigned char const *>(&x);
        cloud.data.push_back(pcx[0]);
        cloud.data.push_back(pcx[1]);
        cloud.data.push_back(pcx[2]);
        cloud.data.push_back(pcx[3]);
        unsigned char const * pcy = reinterpret_cast<unsigned char const *>(&y);
        cloud.data.push_back(pcy[0]);
        cloud.data.push_back(pcy[1]);
        cloud.data.push_back(pcy[2]);
        cloud.data.push_back(pcy[3]);
        unsigned char const * pcz = reinterpret_cast<unsigned char const *>(&z);
        cloud.data.push_back(pcz[0]);
        cloud.data.push_back(pcz[1]);
        cloud.data.push_back(pcz[2]);
        cloud.data.push_back(pcz[3]);
    }
}

size_t findDesc(std::vector<std::pair<KeyFrame*, size_t>>& target_list, std::pair<KeyFrame*, size_t> query){
    size_t re=-1;
    for(int i=0; i<target_list.size(); i++){
        if(query.first==target_list[i].first && query.second==target_list[i].second){
            
            re=i;
            //std::cout<<(int)re<<std::endl;
            break;
        }
    }
    return re;
}

void System::SaveDescTrack(const string &track_file, const string &desc_file, const string &kp_file, const string &posi_file){
    vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();
    std::vector<cv::Mat> posis;
    
    int desc_count=0;
    int track_count=0;
    std::vector<std::pair<KeyFrame*, size_t>> desc_list;
    std::vector<std::vector<size_t>> track_list;
    std::vector<cv::Mat> posi_list;
    for(int i=0; i<vpMPs.size(); i++){
        MapPoint* mp = vpMPs[i];
        if (mp->isBad()){
            continue;
        }
        map<KeyFrame*, size_t> tracks= mp->GetObservations();
        std::vector<size_t> track_out;
        for(auto item: tracks){
            size_t desc_id = findDesc(desc_list, item);
            if(desc_id==-1){
                desc_list.push_back(item);
                track_out.push_back(desc_list.size()-1);
            }else{
                track_out.push_back(desc_id);
            }
        }
        if(track_out.size()>=3){
            track_list.push_back(track_out);
            posi_list.push_back(vpMPs[i]->GetWorldPos());
        }
    }
    std::cout<<"desc_list: "<<desc_list.size()<<std::endl;
    std::cout<<"track_list: "<<track_list.size()<<std::endl;
    ofstream f;
    f.open(track_file.c_str());
    //f << fixed;
    for(int i=0; i<track_list.size(); i++){
        for (auto track: track_list[i]){
            f<<track<<",";
        }
        f<<std::endl;
    }
    f.close();
    f.open(posi_file.c_str());
    for(int i=0; i<posi_list.size(); i++){
        for(int j=0; j<3; j++){
            f<<posi_list[i].at<float>(j)<<",";
        }
        f<<std::endl;
    }
    f.close();
    f.open(kp_file.c_str());
    for(auto desc: desc_list){
        
        cv::KeyPoint kp = desc.first->mvKeysUn[desc.second];
        f<<kp.pt.x<<","<<kp.pt.y<<","<<kp.octave<<","<<desc.first->mnId;
        f<<std::endl;
    }
    f.close();
    f.open(desc_file.c_str());
    for(auto desc: desc_list){
        
        cv::Mat desc_mat = desc.first->mDescriptors.row(desc.second);
        //std::cout<<desc_mat<<std::endl;
        for(int i=0; i<desc_mat.cols; i++){
            f<<(int)desc_mat.at<unsigned char>(0,i)<<",";
        }
        f<<std::endl;
    }
    f.close();
}

void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    
    int count=0;

    for(int i=0; i<vpKFs.size(); i++)
    {
        ORB_SLAM2::KeyFrame* pKF = vpKFs[i];

        while(pKF->isBad())
        {
            continue;
        }

        cv::Mat Trw = pKF->GetPose()*Two;

        cv::Mat Rwc = Trw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Trw.rowRange(0,3).col(3);
        
        std::vector<std::string> splited = split(pKF->file_name_, "/");
        std::string filename= splited.back();
        std::stringstream ss;
        f<<filename<<",";

        f << pKF->mnId<<","<< Rwc.at<float>(0,0) << "," << Rwc.at<float>(0,1)  << "," << Rwc.at<float>(0,2) << ","  << twc.at<float>(0) << "," <<
             Rwc.at<float>(1,0) << "," << Rwc.at<float>(1,1)  << "," << Rwc.at<float>(1,2) << ","  << twc.at<float>(1) << "," <<
             Rwc.at<float>(2,0) << "," << Rwc.at<float>(2,1)  << "," << Rwc.at<float>(2,2) << ","  << twc.at<float>(2) << endl;
        
        count++;
    }
    f.close();
    
    
    rosbag::Bag bag_map;
    rosbag::Bag bag_loc;
    bag_map.open("map.bag", rosbag::bagmode::Write);
    bag_loc.open("loc.bag", rosbag::bagmode::Write);
    sensor_msgs::PointCloud2 cloud;
    vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();
    std::vector<cv::Mat> posis;
    for(int i=0; i<vpMPs.size(); i++){
        ORB_SLAM2::MapPoint* pMP=vpMPs[i];
        if(pMP->isBad()){
            continue;
        }
        posis.push_back(pMP->GetWorldPos());
    }
    convert2MsgCloud(posis, cloud);
    
    bag_map.write<sensor_msgs::PointCloud2>("/map/sparse_point", ros::Time(1), cloud);
    
    int img_count=0;
    geometry_msgs::PoseArray pose_msgs;
    for(int i=0; i<vpKFs.size(); i++)
    {
        ORB_SLAM2::KeyFrame* pKF = vpKFs[i];

        while(pKF->isBad())
        {
            continue;
        }
        cv_bridge::CvImage cv_mat;
        cv::Mat img= cv::imread(pKF->file_name_);
        cv_mat.image=img;
        cv_mat.encoding="bgr8";
        cv_mat.header.seq=img_count;
        cv_mat.header.stamp=ros::Time(1);
        sensor_msgs::CompressedImagePtr img_ptr= cv_mat.toCompressedImageMsg();
        bag_map.write<sensor_msgs::CompressedImage>("/map/image/right", cv_mat.header.stamp, *img_ptr);
        bag_loc.write<sensor_msgs::CompressedImage>("/loc/image", ros::Time(pKF->mnFrameId*0.1+1), *img_ptr);
        vector<MapPoint*> mps = pKF->GetMapPointMatches();
        //
        std::vector<cv::Mat> posis;
        for (int j=0; j<mps.size(); j++){
            ORB_SLAM2::MapPoint* pMP=mps[j];
            //std::cout<<mps[i]<<std::endl;
            if(pMP){
                if (pMP->isBad()){
                    continue;
                }
                //std::cout<<pMP->GetWorldPos().t()<<std::endl;
                posis.push_back(pMP->GetWorldPos());
            }
        }
        sensor_msgs::PointCloud2 cloud;
        
        convert2MsgCloud(posis, cloud);
        bag_loc.write<sensor_msgs::PointCloud2>("/loc/sparse_point",ros::Time(pKF->mnFrameId*0.1+1) ,cloud);

        geometry_msgs::Pose pose_msg;
        cv::Mat pose_inv = pKF->GetPoseInverse();
        float x= pose_inv.at<float>(0, 3);
        float y= pose_inv.at<float>(1, 3);
        float z= pose_inv.at<float>(2, 3);
        pose_msg.position.x=x;
        pose_msg.position.y=y;
        pose_msg.position.z=z;
        
        Eigen::Matrix3f e_mat;
        for(int n=0; n<3; n++){
            for(int m=0; m<3; m++){
                e_mat(n,m)=pose_inv.at<float>(n, m);
            }
        }
        Eigen::Quaternionf qua(e_mat);
        pose_msg.orientation.x=qua.x();
        pose_msg.orientation.y=qua.y();
        pose_msg.orientation.z=qua.z();
        pose_msg.orientation.w=qua.w();
        pose_msgs.poses.push_back(pose_msg);
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.pose=pose_msg;
        pose_stamped.header.frame_id="map";
        bag_loc.write<geometry_msgs::PoseStamped>("/loc/pose", ros::Time(pKF->mnFrameId*0.1+1), pose_stamped);
        
        img_count++;
    }
    pose_msgs.header.frame_id="map";
    bag_map.write<geometry_msgs::PoseArray>("/map/pose", ros::Time(1), pose_msgs);
    bag_map.close();
    bag_loc.close();
    cout << endl << "trajectory saved!" << endl;
}

int System::GetTrackingState()
{
    //unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    //unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    //unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

} //namespace ORB_SLAM
