/**
* Reads a bag file that contains 
* 1) Velodyne raw messages 
* 2) tf messages (e.g. as odometry)
* This class reads the _whole_ tf to cache and uses this to sync the velodyne messages with the motion of the platform. 
* The result is returned as pcl::PointCloud<PointXYZI> in the sensor coordinates
*
* The class uses Velodyne ros pkg, with one hack:

Add the following to rawdata.h:
#include <angles/angles.h>

int setupOffline(std::string calibration_file, double max_range_, double min_range_)
{

    config_.max_range = max_range_;
    config_.min_range = min_range_;
    ROS_INFO_STREAM("data ranges to publish: ["
	<< config_.min_range << ", "
	<< config_.max_range << "]");

    config_.calibrationFile = calibration_file;

    ROS_INFO_STREAM("correction angles: " << config_.calibrationFile);

    calibration_.read(config_.calibrationFile);
    if (!calibration_.initialized) {
	ROS_ERROR_STREAM("Unable to open calibration file: " <<
	    config_.calibrationFile);
	return -1;
    }

    // Set up cached values for sin and cos of all the possible headings
    for (uint16_t rot_index = 0; rot_index < ROTATION_MAX_UNITS; ++rot_index) {
	float rotation = angles::from_degrees(ROTATION_RESOLUTION * rot_index);
	cos_rot_table_[rot_index] = cosf(rotation);
	sin_rot_table_[rot_index] = sinf(rotation);
    }
    return 0;
}

* 
* NOTE: In order for the synchronization of velodyne and vehicle motion to work
* you have to express the vehicle motion in the velodyne sensor coordinates. 
* If your log file only contains odometry or similar for the vehicle you can give 
* an extra link as a parameter (note that then give id to this extra link in constructor in this case). 
*
* @author Jari Saarinen (jari.saarinen@aalto.fi)
*/

#ifndef VELODYNE_BAG_READER_H_
#define VELODYNE_BAG_READER_H_
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <velodyne_pointcloud/rawdata.h>
#include <velodyne_pointcloud/point_types.h>
#include <ndt_offline/PoseInterpolationNavMsgsOdo.h>
#include <iostream>
#include <pcl_ros/impl/transforms.hpp>
#include "pointcloudbagreader.h"
#include "VelodyneBagReader.h"
#ifdef READ_RMLD_MESSAGES
#include<SynchronizedRMLD.h>
#endif


template<class PointT> void convertPointCloud(const velodyne_rawdata::VPointCloud &conv_points,
                       pcl::PointCloud<PointT> &cloud) {
 for(size_t i = 0;i<conv_points.size();i++){
   PointT p;
   p.x =conv_points.points[i].x; p.y=conv_points.points[i].y; p.z=conv_points.points[i].z;
   cloud.push_back(p);
 }
}

template<> void convertPointCloud(const velodyne_rawdata::VPointCloud &conv_points,
                                                       pcl::PointCloud<velodyne_pointcloud::PointXYZIR> &cloud) {
 for(size_t i = 0;i<conv_points.size();i++){
   velodyne_pointcloud::PointXYZIR p;
   p.x =conv_points.points[i].x; p.y=conv_points.points[i].y; p.z=conv_points.points[i].z;
   p.intensity = conv_points.points[i].intensity;
   p.ring = conv_points.points[i].ring;
   cloud.push_back(p);
 }
}


template<class PointT>
class VelodyneBagReader{
    public:
	/**
	 * Constructor
	 * @param calibration_file path and name to your velodyne calibration file
	 * @param bagfilename The path and name of the bagfile you want to handle
	 * @param velodynetopic The topic that contains velodyne_msgs/VelodyneScan
	 * @param tf_pose_id The id of the tf that you want to use
	 * @param fixed_frame_id The name of the fixed frame in tf (default = "/world")
	 * @param tftopic name of tf (default "/tf")
	 * @param dur The buffer size (must be larger than the length of the bag file gives) default = 3600s
	 * @param sensor_link An optional static link that takes e.g. your /odom to the sensor frame 
	 */
	VelodyneBagReader(std::string calibration_file, 
		std::string bagfilename,
		std::string velodynetopic, 
    const std::string &tf_pose_id,
    const std::string &fixed_frame_id="/world",
    const std::string &tftopic="/tf",
		ros::Duration dur = ros::Duration(3600),
		tf::StampedTransform *sensor_link=NULL,
		double velodyne_max_range=130.0, 
		double velodyne_min_range=2.0,
		double sensor_time_offset=0.0) 
      
	{
  	  // The view_direction / view_width is important to have set. The min/max range is overwritten in the setupOffline
	  double view_direction = 0;
	  double view_width = 2*M_PI;
	  dataParser.setParameters(velodyne_min_range,
                                  velodyne_max_range,
                                  view_direction,
                                  view_width);
      std::cout<<"Velodyne PARAMETERS FILTER: max="<<velodyne_max_range<<","<<velodyne_min_range<<std::endl;
      if(dataParser.setupOffline(calibration_file, velodyne_max_range, velodyne_min_range)!=0)
        exit(0);
	    sensor_time_offset_ = ros::Duration(sensor_time_offset);
	    fprintf(stderr,"Opening '%s'\n",bagfilename.c_str());

	    bag.open(bagfilename, rosbag::bagmode::Read);

	    velodynetopic_ = velodynetopic;
	    tf_pose_id_ = tf_pose_id;

	    std::vector<std::string> topics;
	    topics.push_back(tftopic);
	    topics.push_back(velodynetopic_);

#ifdef READ_RMLD_MESSAGES
	    topics.push_back("/rmld/data");
	    topics.push_back("/amtec/tilt_state");
#endif
	    for(int i=0; i<topics.size(); ++i) { 
		fprintf(stderr,"Searched Topic [%d] = '%s'\n",i,topics[i].c_str());
	    }

	    view = new rosbag::View(bag, rosbag::TopicQuery(topics));
	    I = view->begin();

	    odosync = new PoseInterpolationNavMsgsOdo(view,tftopic, fixed_frame_id,dur, sensor_link);
#ifdef READ_RMLD_MESSAGES
	    rmldsync = new SynchronizedRMLD(view,tftopic,"/base_link",dur);
#endif
	    //odosync = NULL;
	}


	/**
	 * Reads the next measurment.
	 * @param cloud The generated point cloud
	 * @param sensor_pose [out] The pose of the sensor origin. (Utilizes the tf_pose_id and sensor_link from the constructor).
	 **/
	bool readNextMeasurement(pcl::PointCloud<PointT> &cloud, tf::Transform &sensor_pose){
	    if(I == view->end()){
		fprintf(stderr,"End of measurement file Reached!!\n");
		return false;
	    }

	    //if(odosync == NULL) return true;
	    rosbag::MessageInstance const m = *I;
	    velodyne_msgs::VelodyneScan::ConstPtr scan = m.instantiate<velodyne_msgs::VelodyneScan>();
	    if (scan != NULL){
		if(m.getTopic() == velodynetopic_){

		    velodyne_rawdata::VPointCloud pnts,conv_points;

		    // process each packet provided by the driver
		    tf::Transform T;
		    ros::Time t0=scan->header.stamp + sensor_time_offset_; 
		    if(odosync->getTransformationForTime(t0, tf_pose_id_, sensor_pose)){
			for (size_t next = 0; next < scan->packets.size(); ++next){
			    dataParser.unpack(scan->packets[next], pnts); // unpack the raw data
			    ros::Time t1=scan->packets[next].stamp + sensor_time_offset_;

			    if(odosync->getTransformationForTime(t0,t1,tf_pose_id_,T)){
				pcl_ros::transformPointCloud(pnts,conv_points,T);
        convertPointCloud(conv_points, cloud);
			    }else{
				//fprintf(stderr,"No transformation\n");
			    }
			    pnts.clear();
			}

		    }else{
			fprintf(stderr,"No transformation\n");
		    }
		    //fprintf(stderr,"Conversion resulted into a %u points\n",cloud.size());					
		}
	    }

	    I++;
	    return true;
	}

	/**
	* @param[in] NMeas : number of point cloud to stack
	* @param[out] cloud<PointT> : point cloud out at the end. pointT is the type of the cloud but it needs to be x,y,z even though it's templated.
	* @param[out] sensor_pose : sensor pose of measurement
	*/
	bool readMultipleMeasurements(unsigned int Nmeas, pcl::PointCloud<PointT> &cloud, tf::Transform &sensor_pose){

	    tf::Transform T;
	    ros::Time t0;
	    velodyne_rawdata::VPointCloud pnts,conv_points;


	    if(!getNextScanMsg()) return false;
	    t0 = global_scan->header.stamp + sensor_time_offset_ ;
	    timestamp_of_last_sensor_message = t0;

	    if(!odosync->getTransformationForTime(t0, tf_pose_id_, sensor_pose)){
		fprintf(stderr,"No Transformation %s for time (%lf)!\n",tf_pose_id_.c_str(), t0.toSec());
		return true; //false for eof
	    }
	    for (size_t next = 0; next < global_scan->packets.size(); ++next){
		ros::Time t1=global_scan->packets[next].stamp + sensor_time_offset_;
		dataParser.unpack(global_scan->packets[next], pnts); // unpack the raw data

		//if(odosync->getTransformationForTime(t0,t1,tf_pose_id_,T)){
		if(odosync->getTransformationForTime(t1,tf_pose_id_,T)){
		    tf::Transform Td = sensor_pose.inverse() * T;

        pcl_ros::transformPointCloud(pnts,conv_points,Td);
        convertPointCloud(conv_points, cloud);
        conv_points.clear();
		}
		pnts.clear();
	    }	

	    for(unsigned int nscans=1;nscans<Nmeas;nscans++){
		if(!getNextScanMsg()) return false;

    for (size_t next = 0; next < global_scan->packets.size(); ++next){
      dataParser.unpack(global_scan->packets[next], pnts); // unpack the raw data
      ros::Time t1=global_scan->packets[next].stamp + sensor_time_offset_;

      //if(odosync->getTransformationForTime(t0,t1,tf_pose_id_,T)){
      if(odosync->getTransformationForTime(t1,tf_pose_id_,T)){
        tf::Transform Td = sensor_pose.inverse() * T;
        pcl_ros::transformPointCloud(pnts,conv_points,Td);
        convertPointCloud(conv_points, cloud);
      }
        pnts.clear();

    }
      }
		//std::cerr<<"Got cloud\n";
		return true;
      }




	    /**
	     * @param[in] NMeas : number of point cloud to stack
	     * @param[out] cloud<PointT> : point cloud out at the end. pointT is the type of the cloud
	     * @param[out] sensor_pose : sensor pose of measurement
	     * @param[out] base_pose : base pose of measurement
	     * @param[in] base_pose_id : base_pose frame name
	     */
    //bool readMultipleMeasurements(unsigned int Nmeas, pcl::PointCloud<PointT> &cloud, tf::Transform &sensor_pose, tf::Transform &base_pose, std::string base_pose_id){
    bool readMultipleMeasurements(unsigned int Nmeas, pcl::PointCloud<PointT> &cloud, tf::Transform &sensor_pose, tf::Transform &base_pose, const std::string &base_pose_id){
			if(readMultipleMeasurements(Nmeas, cloud, sensor_pose )){
				odosync->getTransformationForTime(timestamp_of_last_sensor_message,base_pose_id,base_pose);
			}else{
				return false;
			}
			return true;
	    }

	    /**
	     * Get pose for latest measurement with pose id
	     */
      bool getPoseFor(tf::Transform &pose, const std::string &pose_id){
		if(odosync->getTransformationForTime(timestamp_of_last_sensor_message,pose_id,pose)){
		    return true;
		}
		return false;
	    }



	    bool getNextScanMsg(){
		if(I == view->end()){
		    fprintf(stderr,"End of measurement file Reached!!\n");
		    return false;
		}
		bool done = false; 
		while(!done){
		    rosbag::MessageInstance const m = *I;
		    global_scan = m.instantiate<velodyne_msgs::VelodyneScan>();
		    I++;
		    if(I == view->end()) done = true;
		    if(global_scan != NULL){
                      //                        fprintf(stderr,"GOT %s \n",m.getTopic().c_str());
			done = true;
		    }
		}

		if(I == view->end()) {
                  return false;
                }
		return true;
	    }

            ros::Time getTimeStampOfLastSensorMsg() const {
                 return timestamp_of_last_sensor_message;
            }


		private:
	    PoseInterpolationNavMsgsOdo *odosync;
	    velodyne_rawdata::RawData dataParser;
	    rosbag::Bag bag;
	    rosbag::View *view;
	    rosbag::View::iterator I;
	    std::string velodynetopic_;
	    std::string tf_pose_id_;
	    velodyne_msgs::VelodyneScan::ConstPtr global_scan;
	    ros::Time timestamp_of_last_sensor_message;
	    ros::Duration sensor_time_offset_;
#ifdef READ_RMLD_MESSAGES
	    SynchronizedRMLD *rmldsync;
		public:
	    bool getSynchronizedRMLD(std::string rmld_tf_name, tf::Transform &rmld_pose, double &rmld_val, double &amtec_pos, int &tilt_state) {
		return rmldsync->getDataForTime(timestamp_of_last_sensor_message,rmld_tf_name,rmld_pose,rmld_val,amtec_pos,tilt_state);
	    }
#endif
	    };


#endif

