#ifndef GRAPH_MAP_H
#define GRAPH_MAP_H
#include "graphfactory.h"
#include "graph_map/factor.h"
#include "graph_map/map_type.h"
#include "graph_map/map_node.h"
#include "stdio.h"
#include <iostream>
#include "boost/shared_ptr.hpp"
#include "Eigen/Dense"
#include <stdint.h>
#include <Eigen/StdVector>
#include "visualization/graph_plot.h"
#include "ndt_map/ndt_map.h"
#include "ndt/ndt_map_type.h"
#include "ros/ros.h"
#include "ros/node_handle.h"
#include "boost/serialization/serialization.hpp"
#include "boost/serialization/vector.hpp"

namespace perception_oru{
namespace libgraphMap{
using namespace perception_oru;
typedef std::vector<Eigen::Affine3d,Eigen::aligned_allocator<Eigen::Affine3d> > EigenAffineVector;
class GraphMap{
    
public:
  /*!
   * \brief UpdateGraph is  high level interface to graph class. It allows for automatic selection and creation of mapnodes.
   */
  /*!
   * \brief AutomaticMapInterchange Automatically interchange between or create new map upon certain conditions
   * \param Tnow pre: Pose is described in the local mapNode frame. Post: Pose is described in the current local mapNode frame and is properly updated if a new frame has been selected
   * \param cov contain the movement uncertainty accumulated since last node
   * \param T_world_to_local pre: Not used: Post T_world_to_local contain the mapping from world to the current local mapNode frame.
   */

  virtual void AddMapNode(const MapParamPtr &mapparam,const Eigen::Affine3d &diff,const Matrix6d &cov);//
  MapNodePtr GetCurrentNode();
  MapNodePtr GetPreviousNode();
  uint32_t GraphSize(){return nodes_.size();}
  virtual string ToString();
  virtual NodePtr GetNode(int nodeNr);
  virtual Affine3d GetNodePose(int nodeNr);
  virtual Eigen::Affine3d GetCurrentNodePose();
  virtual Eigen::Affine3d GetPreviousNodePose();
  virtual void AddFactor(MapNodePtr prev, MapNodePtr next, Affine3d Tdiff, Matrix6d cov);


  //virtual std::vector<FactorPtr> GetFactors(NodePtr node);//Get all factors for current node
  GraphMap();
protected:
  GraphMap(const Eigen::Affine3d &nodepose, const MapParamPtr &mapparam, const GraphParamPtr graphparam);
  MapNodePtr currentNode_=NULL,prevNode_=NULL;//The current node
  std::vector<NodePtr> nodes_;//Vector of all nodes in graph
  std::vector<FactorPtr> factors_;
  std::vector<MapNodePtr> map_nodes_;
  MapParamPtr mapparam_=NULL;//
  bool use_submap_=false;
  double interchange_radius_=0;
  double compound_radius_=0;
  double min_keyframe_dist_=0.5;
  double min_keyframe_rot_deg_=15;
  bool use_keyframe_=true;
private:
  friend class GraphFactory;
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & currentNode_;
    ar & prevNode_;
    ar & nodes_;
    ar & map_nodes_;
    ar & factors_;
    ar & use_submap_;
    ar & interchange_radius_;
    ar & compound_radius_;
  }

};


class GraphParam{
public:
  void GetParametersFromRos();
  bool use_submap_=false;
  double interchange_radius_=0;
  double compound_radius_=0;
  bool use_keyframe_=true;
  double min_keyframe_dist_=0.5;
  double min_keyframe_rot_deg_=15;
  GraphParam();
private:
  friend class GraphFactory;
  /* friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
  }*/
};
}
}
#endif // GRAPH_H
