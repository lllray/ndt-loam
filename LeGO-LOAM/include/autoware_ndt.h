//
// Created by ray on 20-4-19.
//

#ifndef LEGO_LOAM_AUTOWARE_NDT_H
#define LEGO_LOAM_AUTOWARE_NDT_H

#include <pcl/io/io.h>

typedef pcl::PointXYZI  PointType;
void autoware_ndt_init();
void autoware_ndt_match(pcl::PointCloud<PointType>::Ptr cloudIn,float T[6]);
pcl::PointCloud<pcl::PointXYZ>  autoware_ndt_match_step(pcl::PointCloud<PointType>::Ptr cloudIn,float T[6]);
#endif //LEGO_LOAM_AUTOWARE_NDT_H
