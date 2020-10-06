//
// Created by ray on 20-2-28.
//

#ifndef LEGO_LOAM_NDTMAP_H
#define LEGO_LOAM_NDTMAP_H

typedef pcl::PointXYZI  PointType;
void ndt_init();
pcl::PointCloud<pcl::PointXYZ>  ndt_match_step(pcl::PointCloud<PointType>::Ptr cloudIn,float T[6]);
void ndt_match(pcl::PointCloud<PointType>::Ptr cloudIn,float T[6]);
#endif //LEGO_LOAM_NDTMAP_H
