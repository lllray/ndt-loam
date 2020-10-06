//
// Created by ray on 20-2-28.
//


#include <ndt_map_builder/ndt_map_builder.h>
//#include <ndt_map/oc_tree.h>
#include <ndt_registration/ndt_matcher_d2d.h>
#include <ndt_map/pointcloud_utils.h>
#include <pcl/io/pcd_io.h>

#include "ndtMap.h"


using namespace std;
using namespace perception_oru;

#define PI 3.1415926
//double __res2[] = {0.2,0.5,1,2,5};
double __res2[] = {1,4};
//double __res2[] = {0.2};

double res=0.5;

std::vector<double> resolutions=std::vector<double>(__res2, __res2+sizeof(__res2)/sizeof(double));;

double MAX_DIST = 50;

perception_oru::NDTMatcherD2D matcherF2F(false, false, resolutions);
perception_oru::NDTMatcherP2D matcherP2F(resolutions);

bool doHistogram=true;


bool type_d2d=true;

NDTMapBuilder mapper(res,doHistogram);

void ndt_init(){

    if(type_d2d)
    {
        mapper.setMatcherF2F(&matcherF2F);
        cout<<"setting to D2D matcher\n";
    }
    else //p2d
    {
        mapper.setMatcherP2F(&matcherP2F);
        cout<<"setting to P2D matcher\n";
    }
}
//
pcl::PointCloud<pcl::PointXYZ>  ndt_match_step(pcl::PointCloud<PointType>::Ptr cloudIn,float T[6]){

    pcl::PointCloud<pcl::PointXYZ> cloudTemp;

    int cloudSize=cloudIn->points.size();

    cout << "[ndtMap]cloudIn size is:"<< cloudSize << endl;
    for (int i = 0; i < cloudSize; i++) {


        double dist = sqrt(pow(cloudIn->points[i].x,2)+pow(cloudIn->points[i].y,2)+pow(cloudIn->points[i].z,2));
        if(dist<MAX_DIST)
        {
            pcl::PointXYZ pNew;

            pNew.x=cloudIn->points[i].z;
            pNew.y=cloudIn->points[i].x;
            pNew.z=-cloudIn->points[i].y;

            cloudTemp.points.push_back(pNew);
        }else {
            //cout << "[ndtMap]there is a point out MAX_DIST" << endl;
        }

    }
  cout << "[ndtMap] cloudTemp size is:"<< cloudTemp.points.size() << endl;

    /*
    lx debug 0301
    [ndpMap]  T_test is T  0.2 -0.2  0.1 r -0  0 -0
    Matching scans with ids 0 and 1
    [ndt_map_builder] T fin 0 0 0 r 9.84055e-05  0.00169473  -0.0202203
    T init 0 0 0 r 9.84055e-05  0.00169473  -0.0202203
    score = 0best is 0
    T fin  -0.200334   0.199988 -0.0997561 r  3.14159 -3.14158  3.14159
     */
    pcl::PointCloud<pcl::PointXYZ> cloud_test;
    Eigen::Transform<double,3,Eigen::Affine,Eigen::ColMajor> T_test;
    T_test.setIdentity();

    T_test =  Eigen::Translation<double,3>(0.2,0.1,0.3)*
    Eigen::AngleAxis<double>(0.2,Eigen::Vector3d::UnitX()) *
    Eigen::AngleAxis<double>(0.3,Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxis<double>(0.5,Eigen::Vector3d::UnitZ()) ;

    static bool first=false;
    Eigen::Transform<double,3,Eigen::Affine,Eigen::ColMajor> T_out;
    if(first) {
        std::cout << "[ndpMap]  T_test is " << "T " << T_test.translation().transpose()
                << " r " << T_test.rotation().eulerAngles(0, 1, 2).transpose() << std::endl;
                 // << " r " << T_test.rotation().inverse().eulerAngles(0, 1, 2).transpose() << std::endl;
        cloud_test = perception_oru::transformPointCloud(T_test, cloudTemp);
        T_out=mapper.addScan(cloud_test);
        std::cout << "[ndpMap]  T_out is " << "T " << T_out.translation().transpose()
                  << " r " << T_out.rotation().eulerAngles(0, 1, 2).transpose() << std::endl;
        //T_out.translation().transpose()[0]=T_out.translation().transpose()[0];
        return perception_oru::transformPointCloud(T_out, cloud_test);

    }else {
        T_out=mapper.addScan(cloudTemp);
        first=true;
        return cloudTemp;
    }
}

void ndt_match(pcl::PointCloud<PointType>::Ptr cloudIn,float T[6]){
    pcl::PointCloud<pcl::PointXYZ> cloudTemp;
    int cloudSize=cloudIn->points.size();
    for (int i = 0; i < cloudSize; i++) {
        double dist = sqrt(pow(cloudIn->points[i].x,2)+pow(cloudIn->points[i].y,2)+pow(cloudIn->points[i].z,2));
        if(dist<MAX_DIST)
        {
            pcl::PointXYZ pNew;
            pNew.x=cloudIn->points[i].z;
            pNew.y=cloudIn->points[i].x;
            pNew.z=-cloudIn->points[i].y;
//            pNew.x=cloudIn->points[i].x;
//            pNew.y=cloudIn->points[i].y;
//            pNew.z=cloudIn->points[i].z;

            cloudTemp.points.push_back(pNew);
        }else {
        }
    }
    //cout << "[ndtMap] cloudTemp size is:"<< cloudTemp.points.size() << endl;
    Eigen::Transform<double,3,Eigen::Affine,Eigen::ColMajor> T_out;

    /*增加先验*/
    pcl::PointCloud<pcl::PointXYZ> cloud_test;
    Eigen::Transform<double,3,Eigen::Affine,Eigen::ColMajor> T_test;
    T_test.setIdentity();

    T_test =  Eigen::Translation<double,3>(0,0,T[4])*
    Eigen::AngleAxis<double>(T[0],Eigen::Vector3d::UnitX()) *
    Eigen::AngleAxis<double>(T[2],Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxis<double>(0.0,Eigen::Vector3d::UnitZ()) ;
    cloud_test = perception_oru::transformPointCloud(T_test, cloudTemp);
    T_out=mapper.addScan(cloud_test);

   // T_out=mapper.addScan(cloudTemp);
    std::cout << "[ndpMap]  T_out is " << "T " << T_out.translation().transpose()
              << " r " << T_out.rotation().eulerAngles(0, 1, 2).transpose() << std::endl;
    T[1] = -T_out.rotation().eulerAngles(0, 1, 2).transpose()[2];
    if(T[1]>1.7)T[1]=(PI-T[1]);
    else if(T[1]<-1.7)T[1]=(-PI-T[1]);
    T[3] = T_out.translation().transpose()[0];
    T[5] = T_out.translation().transpose()[1];

}//旋转角可能方向正确，但是角度太小
//Ｔ R 均以对齐