#ifndef GRAPH_NAVIGATOR_INTERFACE_H
#define GRAPH_NAVIGATOR_INTERFACE_H
#include "graph_map/graph_map.h"
#include "graph_map/map_node.h"
#include "graph_map/map_type.h"
#include "graphfactory.h"
#include "Eigen/Dense"
#include <boost/serialization/base_object.hpp>
#include "boost/serialization/serialization.hpp"
#include <boost/serialization/export.hpp>
#include <boost/serialization/version.hpp>
#include "boost/serialization/shared_ptr.hpp"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace perception_oru{
namespace libgraphMap{

using namespace perception_oru;
class GraphMapNavigator:public GraphMap{
public:
  GraphMapNavigator(const Eigen::Affine3d &nodepose, const MapParamPtr &mapparam, const GraphParamPtr graphparam);
  GraphMapNavigator(){}
  //!
  //! \brief SwitchToClosestMapNode Attempts to find the closest Mapnode within desired radius
  //! \param Tnow is the position around the closest map will be found
  //! \param cov not used
  //! \param T_world_to_local_map not used
  //! \param radius the maximum distance to search within, 0.0 allows nodes at any distance
  //! \return true if a node was found within range
  bool SwitchToClosestMapNode(Affine3d &Tnow, const Matrix6d &cov, Affine3d & T_world_to_local_map,const double radius);
  //!
  //! \brief SwitchToClosestMapNode returns the closest map node, return true if a node was fund
  //! \param Tnow is the target pose in the global frame
  //! \return true if a node was found
  //!
  bool SwitchToClosestMapNode(Affine3d &Tnow);
  //!
  //! \brief SwitchToClosestMapNode
  //! \param Tnow  is the target pose in the global frame
  //! \param new_current After the method is invoked, new_current will contain true if a new map node was selected as(closest )map node and assigned to current_node;
  //! \return
  //!
  bool SwitchToClosestMapNode(Affine3d &Tnow, bool &new_current);
  bool AutomaticMapInterchange(Affine3d &Tnow, const Matrix6d &cov, Affine3d & T_world_to_local_map,bool &changed_map_node,bool &created_map_node);
  string ToString();
private:

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)//In order to clal this you need to register it to boost using "ar.template register_type<LazyGrid>();"
  {
     ar & boost::serialization::base_object<GraphMap>(*this);
  }

};
void LoadGraphMap(const string &file_name,  GraphMapNavigatorPtr ptr);



}
}
#endif // GRAPH_NAVIGATOR_INTERFACE_H
