#include "graph_map/ndt_dl/ndtdl_map_type.h"
#include <boost/serialization/export.hpp>
#include <graph_map/ndt_dl/point_curv.h>

BOOST_CLASS_EXPORT(perception_oru::libgraphMap::NDTDLMapType)
namespace perception_oru{
namespace libgraphMap{
  using namespace std;


  NDTDLMapType::NDTDLMapType( MapParamPtr paramptr) : MapType(paramptr){
    NDTDLMapParamPtr param = boost::dynamic_pointer_cast< NDTDLMapParam >(paramptr);//Should not be NULL
    if(param!=NULL){
      resolution_=param ->resolution_;
      map_flat_ = new perception_oru::NDTMap(new perception_oru::LazyGrid(resolution_));
      map_flat_->initialize(0.0,0.0,0.0,param->sizex_,param->sizey_,param->sizez_);
      map_edge_ = new perception_oru::NDTMap(new perception_oru::LazyGrid(resolution_));
      map_edge_->initialize(0.0,0.0,0.0,param->sizex_,param->sizey_,param->sizez_);
      cout<<"created ndtdlmap"<<endl;
    }
    else
      cerr<<"templateMapType: Cannot create instance for \"templateMapType\""<<std::endl;
  }
  NDTDLMapType::~NDTDLMapType(){}

  void NDTDLMapType::update(const Eigen::Affine3d &Tsensor,pcl::PointCloud<pcl::PointXYZ> &cloud, bool simple){//update map, cloud is the scan, Tsensor is the pose where the scan was aquired.

    cout<<"The NDT-DL update with PointXYZ is please implement map update for NDT-DL"<<endl;
    if(initialized_){
      //Initialize map
    }else{
      //Update map
      initialized_ = true;
    }
  }

  void NDTDLMapType::update(const Eigen::Affine3d &Tsensor,pcl::PointCloud<velodyne_pointcloud::PointXYZIR> &cloud, bool simple){//update map, cloud is the scan, Tsensor is the pose where the scan was aquired.


    // Segment the point based on curvature


    pcl::PointCloud<pcl::PointXYZ> cornerPointsSharp;
    pcl::PointCloud<pcl::PointXYZ> cornerPointsLessSharp;
    pcl::PointCloud<pcl::PointXYZ> surfPointsFlat;
    pcl::PointCloud<pcl::PointXYZ> surfPointsLessFlat;

    segmentPointCurvature(cloud, cornerPointsSharp, cornerPointsLessSharp, surfPointsFlat, surfPointsLessFlat);
    // Add the different point clouds into different maps for now only use the flat ones.

    ROS_INFO_STREAM("flatpoints size : " << surfPointsLessFlat.size());
    ROS_INFO_STREAM("edgepoints size : " << cornerPointsLessSharp.size());


    if(initialized_ && enable_mapping_){
      Eigen::Vector3d localMapSize(max_range_,max_range_,sizez_);
      map_flat_->addPointCloudMeanUpdate(Tsensor.translation(),surfPointsLessFlat,localMapSize, 1e5, 25, 2*sizez_, 0.06);
      map_edge_->addPointCloudMeanUpdate(Tsensor.translation(),cornerPointsLessSharp,localMapSize, 1e5, 25, 2*sizez_, 0.06);
    }
    else if(!initialized_){
      InitializeMap(Tsensor,surfPointsLessFlat, cornerPointsSharp);
      initialized_ = true;
    }
  }

  void NDTDLMapType::InitializeMap(const Eigen::Affine3d &Tsensor,pcl::PointCloud<pcl::PointXYZ> &cloudFlat, pcl::PointCloud<pcl::PointXYZ> &cloudEdge){
    cout<<"initialize map"<<endl;
    map_flat_->addPointCloud(Tsensor.translation(),cloudFlat, 0.1, 100.0, 0.1);
    map_flat_->computeNDTCells(CELL_UPDATE_MODE_SAMPLE_VARIANCE, 1e5, 255, Tsensor.translation(), 0.1);

    map_edge_->addPointCloud(Tsensor.translation(),cloudEdge, 0.1, 100.0, 0.1);
    map_edge_->computeNDTCells(CELL_UPDATE_MODE_SAMPLE_VARIANCE, 1e5, 255, Tsensor.translation(), 0.1);
  }

  bool NDTDLMapType::CompoundMapsByRadius(MapTypePtr target,const Affine3d &T_source,const Affine3d &T_target, double radius){

    return true;
    cout<<"please implement map compound for improved usage of submaps"<<endl;
    if( NDTDLMapPtr targetPtr=boost::dynamic_pointer_cast<NDTDLMapType>(target) ){

      cout<<"\"CompoundMapsByRadius\" not overrided by template but not implemented"<<endl;
    }
  }



}
}

