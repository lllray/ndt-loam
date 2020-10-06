#include "ros/ros.h"
#include <ndt_map/ndt_conversions.h>
#include <ndt_map/ndt_map.h>
#include <ndt_map/ndt_cell.h>
#include <ndt_map/lazy_grid.h>
#include <ndt_map/pointcloud_utils.h>

#include <ndt_map/NDTMapMsg.h>

#include "pcl/point_cloud.h"
#include "pcl/io/pcd_io.h"
#include "pcl/features/feature.h"
#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char** argv){
  ros::init(argc,argv,"map_topic");
  ros::NodeHandle nh;
  ros::Publisher map_pub = nh.advertise<ndt_map::NDTMapMsg>("dummy_map_pub", 1000);
  ros::Rate loop_rate(1);
  ndt_map::NDTMapMsg msg;

  perception_oru::NDTMap nd(new perception_oru::LazyGrid(0.4));
  ROS_INFO("loading from jff...\n");
  if (nd.loadFromJFF("basement_04m.1.jff") < 0)
    ROS_INFO("loading from jff failed\n");
  
  perception_oru::toMessage(&nd,msg,"base");
  while (ros::ok()){
    map_pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}
