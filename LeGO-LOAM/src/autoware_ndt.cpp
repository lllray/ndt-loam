//
// Created by ray on 20-4-19.
//
#define USE_PCL_OPENMP
#include "autoware_ndt.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/ndt.h>
#include <tf/transform_datatypes.h>

#ifdef USE_PCL_OPENMP
#include <pcl_omp_registration/ndt.h>
#endif
struct pose
{
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
};
enum class MethodType
{
    PCL_GENERIC = 0,
    PCL_ANH = 1,
    PCL_ANH_GPU = 2,
    PCL_OPENMP = 3,
};
static MethodType _method_type = MethodType::PCL_OPENMP;
static int initial_scan_loaded = 0;
static pcl::PointCloud<pcl::PointXYZI> map;
static double voxel_leaf_size = 2.0;
// Default values
static int max_iter = 30;        // Maximum iterations
static float ndt_res = 1.0;      // Resolution
static double step_size = 0.1;   // Step size
static double trans_eps = 0.01;  // Transformation epsilon
static pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
#ifdef USE_PCL_OPENMP
static pcl_omp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> omp_ndt;
#endif
static double fitness_score;
static bool has_converged;
static int final_num_iteration;
static double transformation_probability;
static pose previous_pose, guess_pose, guess_pose_imu, guess_pose_odom, guess_pose_imu_odom, current_pose,
        current_pose_imu, current_pose_odom, current_pose_imu_odom, ndt_pose, added_pose, localizer_pose;
static double min_scan_range = 5.0;
static double max_scan_range = 200.0;
static double min_add_scan_shift = 1.0;
static Eigen::Matrix4f tf_btol, tf_ltob;
static double _tf_x=0, _tf_y=0, _tf_z=0, _tf_roll=0, _tf_pitch=0, _tf_yaw=0;
void autoware_ndt_init(){
    Eigen::Translation3f tl_btol(_tf_x, _tf_y, _tf_z);                 // tl: translation
    Eigen::AngleAxisf rot_x_btol(_tf_roll, Eigen::Vector3f::UnitX());  // rot: rotation
    Eigen::AngleAxisf rot_y_btol(_tf_pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rot_z_btol(_tf_yaw, Eigen::Vector3f::UnitZ());
    tf_btol = (tl_btol * rot_z_btol * rot_y_btol * rot_x_btol).matrix();
    tf_ltob = tf_btol.inverse();
}
void autoware_ndt_match(pcl::PointCloud<PointType>::Ptr cloudIn,float T[6]) {
    std::cout << "[ndt_mapping] ------------------------------------" << std::endl;
    //std::cout<<"[ndt_mapping] strat process points"<<std::endl;
    double r;//点云到车的距离
    pcl::PointXYZI p;
    pcl::PointCloud<pcl::PointXYZI> scan;
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    Eigen::Matrix4f t_localizer(Eigen::Matrix4f::Identity());
    Eigen::Matrix4f t_base_link(Eigen::Matrix4f::Identity());

    // Add initial point cloud to velodyne_map
    // 如果点云未曾载入，将第一帧点云加入至地图
    for (pcl::PointCloud<pcl::PointXYZI>::const_iterator item = cloudIn->begin(); item != cloudIn->end(); item++) {
        p.x = (double) item->z;
        p.y = (double) item->x;
        p.z = -(double) item->y;
        p.intensity = (double) item->intensity;

        r = sqrt(pow(p.x, 2.0) + pow(p.y, 2.0));
        //距离阈值 5m——200m
        if (min_scan_range < r && r < max_scan_range) {
            scan.push_back(p);
        }
    }
    pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZI>(scan));
    if (initial_scan_loaded == 0) {
        //std::cout<<"[ndt_mapping] initial_scan_loaded"<<std::endl;
        //点云　从雷达系到世界系
        pcl::transformPointCloud(*scan_ptr, *transformed_scan_ptr, tf_btol);
        map += *scan_ptr;
        initial_scan_loaded = 1;
    }

    // Apply voxelgrid filter
    // 对点云进行体素滤波　scan_ptr->filtered_scan_ptr
    //std::cout<<"[ndt_mapping] Apply voxelgrid filter"<<std::endl;
    pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
    voxel_grid_filter.setInputCloud(scan_ptr);
    voxel_grid_filter.filter(*filtered_scan_ptr);

    pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr(new pcl::PointCloud<pcl::PointXYZI>(map));
//ndt 参数设置
    if (_method_type == MethodType::PCL_GENERIC) {
        //std::cout<<"[ndt_mapping] PCL_GENERIC ndt set"<<std::endl;

        ndt.setTransformationEpsilon(trans_eps);
        ndt.setStepSize(step_size);
        ndt.setResolution(ndt_res);
        ndt.setMaximumIterations(max_iter);
        ndt.setInputSource(filtered_scan_ptr);
    }
#ifdef USE_PCL_OPENMP
    else if (_method_type == MethodType::PCL_OPENMP) {
        omp_ndt.setTransformationEpsilon(trans_eps);
        omp_ndt.setStepSize(step_size);
        omp_ndt.setResolution(ndt_res);
        omp_ndt.setMaximumIterations(max_iter);
        omp_ndt.setInputSource(filtered_scan_ptr);
    }

#endif
//设置ndt算法的参考帧
//如果第一帧，将第一帧设为目标点云

    static bool is_first_map = true;
    if (is_first_map == true) {
        //std::cout<<"[ndt_mapping] is_first_map"<<std::endl;
        if (_method_type == MethodType::PCL_GENERIC)
            ndt.setInputTarget(map_ptr);
#ifdef USE_PCL_OPENMP
        else if (_method_type == MethodType::PCL_OPENMP)
            omp_ndt.setInputTarget(map_ptr);
#endif
        is_first_map = false;
    }

    //无传感器情况下，用上一帧与上上帧的差值作为diff
    guess_pose.x = previous_pose.x+ T[3];
    guess_pose.y = previous_pose.y+T[5];
    guess_pose.z = previous_pose.z + T[4];
    guess_pose.roll = previous_pose.roll - T[0];
    guess_pose.pitch = previous_pose.pitch - T[2];
    guess_pose.yaw = previous_pose.yaw -T[1];


    Eigen::AngleAxisf init_rotation_x(guess_pose.roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf init_rotation_y(guess_pose.pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf init_rotation_z(guess_pose.yaw, Eigen::Vector3f::UnitZ());

    Eigen::Translation3f init_translation(guess_pose.x, guess_pose.y, guess_pose.z);

    //初始位姿矩阵　4*4
    //body位姿->laser位姿
    Eigen::Matrix4f init_guess =
            (init_translation * init_rotation_z * init_rotation_y * init_rotation_x).matrix() * tf_btol;


    pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    //std::cout<<"[ndt_mapping] start matching"<<std::endl;
    if (_method_type == MethodType::PCL_GENERIC) {
        //进行匹配运算
        ndt.align(*output_cloud, init_guess);
        fitness_score = ndt.getFitnessScore();
        //返回final_transformation_
        t_localizer = ndt.getFinalTransformation();
        //返回一个bool变量
        has_converged = ndt.hasConverged();
        //返回迭代次数
        final_num_iteration = ndt.getFinalNumIteration();
        transformation_probability = ndt.getTransformationProbability();
    }
#ifdef USE_PCL_OPENMP
    else if (_method_type == MethodType::PCL_OPENMP) {
        omp_ndt.align(*output_cloud, init_guess);
        fitness_score = omp_ndt.getFitnessScore();
        t_localizer = omp_ndt.getFinalTransformation();
        has_converged = omp_ndt.hasConverged();
        final_num_iteration = omp_ndt.getFinalNumIteration();
    }
#endif

    //估计出的位姿
    //laser位姿->body位姿
    t_base_link = t_localizer * tf_ltob;

    //点云加入地图
    //通过laser位姿
    pcl::transformPointCloud(*scan_ptr, *transformed_scan_ptr, t_localizer);

    tf::Matrix3x3 mat_l, mat_b;

    mat_l.setValue(static_cast<double>(t_localizer(0, 0)), static_cast<double>(t_localizer(0, 1)),
                   static_cast<double>(t_localizer(0, 2)), static_cast<double>(t_localizer(1, 0)),
                   static_cast<double>(t_localizer(1, 1)), static_cast<double>(t_localizer(1, 2)),
                   static_cast<double>(t_localizer(2, 0)), static_cast<double>(t_localizer(2, 1)),
                   static_cast<double>(t_localizer(2, 2)));

    mat_b.setValue(static_cast<double>(t_base_link(0, 0)), static_cast<double>(t_base_link(0, 1)),
                   static_cast<double>(t_base_link(0, 2)), static_cast<double>(t_base_link(1, 0)),
                   static_cast<double>(t_base_link(1, 1)), static_cast<double>(t_base_link(1, 2)),
                   static_cast<double>(t_base_link(2, 0)), static_cast<double>(t_base_link(2, 1)),
                   static_cast<double>(t_base_link(2, 2)));

    // Update localizer_pose.
    localizer_pose.x = t_localizer(0, 3);
    localizer_pose.y = t_localizer(1, 3);
    localizer_pose.z = t_localizer(2, 3);
    mat_l.getRPY(localizer_pose.roll, localizer_pose.pitch, localizer_pose.yaw, 1);

    // Update ndt_pose.
    ndt_pose.x = t_base_link(0, 3);
    ndt_pose.y = t_base_link(1, 3);
    ndt_pose.z = t_base_link(2, 3);
    mat_b.getRPY(ndt_pose.roll, ndt_pose.pitch, ndt_pose.yaw, 1);

    //当前位置　map系
    current_pose.x = ndt_pose.x;
    current_pose.y = ndt_pose.y;
    current_pose.z = ndt_pose.z;
    current_pose.roll = ndt_pose.roll;
    current_pose.pitch = ndt_pose.pitch;
    current_pose.yaw = ndt_pose.yaw;

    T[3] = current_pose.x - previous_pose.x;
    T[5] = current_pose.y - previous_pose.y;
    T[4] = current_pose.z - previous_pose.z;
    T[0] = -(current_pose.roll - previous_pose.roll);
    T[2] = -(current_pose.pitch - previous_pose.pitch);
    T[1] = -(current_pose.yaw - previous_pose.yaw);


    // Update position and posture. current_pos -> previous_pos
    previous_pose.x = current_pose.x;
    previous_pose.y = current_pose.y;
    previous_pose.z = current_pose.z;
    previous_pose.roll = current_pose.roll;
    previous_pose.pitch = current_pose.pitch;
    previous_pose.yaw = current_pose.yaw;


    // Calculate the shift between added_pos and current_pos
    //计算当前位姿与上一次加入地图的位姿是否大于一定值,主要通过平移距离
    double shift = sqrt(pow(current_pose.x - added_pose.x, 2.0) + pow(current_pose.y - added_pose.y, 2.0));
    if (shift >= min_add_scan_shift) {
        //std::cout<<"[ndt_mapping] add to map"<<std::endl;
        map += *transformed_scan_ptr;
        added_pose.x = current_pose.x;
        added_pose.y = current_pose.y;
        added_pose.z = current_pose.z;
        added_pose.roll = current_pose.roll;
        added_pose.pitch = current_pose.pitch;
        added_pose.yaw = current_pose.yaw;

        if (_method_type == MethodType::PCL_GENERIC)
            ndt.setInputTarget(map_ptr);
#ifdef USE_PCL_OPENMP
        else if (_method_type == MethodType::PCL_OPENMP)
            omp_ndt.setInputTarget(map_ptr);
#endif
    }

    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "Sequence number: " << cloudIn->header.seq << std::endl;
    std::cout << "Number of scan points: " << scan_ptr->size() << " points." << std::endl;
    std::cout << "Number of filtered scan points: " << filtered_scan_ptr->size() << " points." << std::endl;
    std::cout << "transformed_scan_ptr: " << transformed_scan_ptr->points.size() << " points." << std::endl;
    std::cout << "map: " << map.points.size() << " points." << std::endl;
    std::cout << "NDT has converged: " << has_converged << std::endl;
    std::cout << "Fitness score: " << fitness_score << std::endl;
    std::cout << "Number of iteration: " << final_num_iteration << std::endl;
    std::cout << "(x,y,z,roll,pitch,yaw):" << std::endl;
    std::cout << "(" << current_pose.x << ", " << current_pose.y << ", " << current_pose.z << ", " << current_pose.roll
              << ", " << current_pose.pitch << ", " << current_pose.yaw << ")" << std::endl;
    std::cout << "Transformation Matrix:" << std::endl;
    std::cout << t_localizer << std::endl;
    std::cout << "shift: " << shift << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;
}
//TODO １,分两次匹配 2,只使用非地面点　3 加速

pcl::PointCloud<pcl::PointXYZ>  autoware_ndt_match_step(pcl::PointCloud<PointType>::Ptr cloudIn,float T[6]) {

    std::cout << "[ndt_mapping] ------------------------------------" << std::endl;
    //std::cout<<"[ndt_mapping] strat process points"<<std::endl;
    double r;//点云到车的距离
    pcl::PointXYZI p;
    pcl::PointCloud<pcl::PointXYZI> scan;
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    Eigen::Matrix4f t_localizer(Eigen::Matrix4f::Identity());
    Eigen::Matrix4f t_base_link(Eigen::Matrix4f::Identity());

    // Add initial point cloud to velodyne_map
    // 如果点云未曾载入，将第一帧点云加入至地图
    for (pcl::PointCloud<pcl::PointXYZI>::const_iterator item = cloudIn->begin(); item != cloudIn->end(); item++) {
        p.x = (double) item->z;
        p.y = (double) item->x;
        p.z = -(double) item->y;
        p.intensity = (double) item->intensity;

        r = sqrt(pow(p.x, 2.0) + pow(p.y, 2.0));
        //距离阈值 5m——200m
        if (min_scan_range < r && r < max_scan_range) {
            scan.push_back(p);
        }
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZI>(scan));
    pcl::PointCloud<pcl::PointXYZI>::Ptr test_scan_ptr;
    if (initial_scan_loaded != 0) {
        Eigen::Matrix4f t_test(Eigen::Matrix4f::Identity());
        Eigen::Translation3f tl_btol(0.2, 0.1, 0.0);                 // tl: translation
        Eigen::AngleAxisf rot_x_btol(0.0, Eigen::Vector3f::UnitX());  // rot: rotation
        Eigen::AngleAxisf rot_y_btol(0.0, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf rot_z_btol(0.15, Eigen::Vector3f::UnitZ());
        t_test = (tl_btol * rot_z_btol * rot_y_btol * rot_x_btol).matrix();
        pcl::transformPointCloud(*scan_ptr, *scan_ptr, t_test);
        std::cout << "test tf t" << tl_btol.x() << " " << tl_btol.y() << " " << tl_btol.z() << std::endl;
        std::cout << "test tf R" << rot_x_btol.angle() << " " << rot_y_btol.angle() << " " << rot_z_btol.angle()
                  << std::endl;
    }
    if (initial_scan_loaded == 0) {
        //std::cout<<"[ndt_mapping] initial_scan_loaded"<<std::endl;
        //点云　从雷达系到世界系
        pcl::transformPointCloud(*scan_ptr, *transformed_scan_ptr, tf_btol);
        map += *scan_ptr;
        initial_scan_loaded = 1;
    }

    // Apply voxelgrid filter
    // 对点云进行体素滤波　scan_ptr->filtered_scan_ptr
    //std::cout<<"[ndt_mapping] Apply voxelgrid filter"<<std::endl;
    pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);

    voxel_grid_filter.setInputCloud(scan_ptr);

    voxel_grid_filter.filter(*filtered_scan_ptr);

    pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr(new pcl::PointCloud<pcl::PointXYZI>(map));
//ndt 参数设置


    if (_method_type == MethodType::PCL_GENERIC) {
        //std::cout<<"[ndt_mapping] PCL_GENERIC ndt set"<<std::endl;

        ndt.setTransformationEpsilon(trans_eps);
        ndt.setStepSize(step_size);
        ndt.setResolution(ndt_res);
        ndt.setMaximumIterations(max_iter);
        ndt.setInputSource(filtered_scan_ptr);
    }
#ifdef USE_PCL_OPENMP
    else if (_method_type == MethodType::PCL_OPENMP) {
        omp_ndt.setTransformationEpsilon(trans_eps);
        omp_ndt.setStepSize(step_size);
        omp_ndt.setResolution(ndt_res);
        omp_ndt.setMaximumIterations(max_iter);
        omp_ndt.setInputSource(filtered_scan_ptr);
    }

#endif
//设置ndt算法的参考帧
//如果第一帧，将第一帧设为目标点云

    static bool is_first_map = true;
    if (is_first_map == true) {
        //std::cout<<"[ndt_mapping] is_first_map"<<std::endl;
        if (_method_type == MethodType::PCL_GENERIC)
            ndt.setInputTarget(map_ptr);
#ifdef USE_PCL_OPENMP
        else if (_method_type == MethodType::PCL_OPENMP)
            omp_ndt.setInputTarget(map_ptr);
#endif
        is_first_map = false;
    }

    //无传感器情况下，用上一帧与上上帧的差值作为diff
    guess_pose.x = previous_pose.x + T[3];
    guess_pose.y = previous_pose.y + T[5];
    guess_pose.z = previous_pose.z + T[4];
    guess_pose.roll = previous_pose.roll + T[0];
    guess_pose.pitch = previous_pose.pitch + T[2];
    guess_pose.yaw = previous_pose.yaw + T[1];


    Eigen::AngleAxisf init_rotation_x(guess_pose.roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf init_rotation_y(guess_pose.pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf init_rotation_z(guess_pose.yaw, Eigen::Vector3f::UnitZ());

    Eigen::Translation3f init_translation(guess_pose.x, guess_pose.y, guess_pose.z);

    //初始位姿矩阵　4*4
    //body位姿->laser位姿
    Eigen::Matrix4f init_guess =
            (init_translation * init_rotation_z * init_rotation_y * init_rotation_x).matrix() * tf_btol;


    pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    //std::cout<<"[ndt_mapping] start matching"<<std::endl;
    if (_method_type == MethodType::PCL_GENERIC) {
        //进行匹配运算
        ndt.align(*output_cloud, init_guess);
        fitness_score = ndt.getFitnessScore();
        //返回final_transformation_
        t_localizer = ndt.getFinalTransformation();
        //返回一个bool变量
        has_converged = ndt.hasConverged();
        //返回迭代次数
        final_num_iteration = ndt.getFinalNumIteration();
        transformation_probability = ndt.getTransformationProbability();
    }
#ifdef USE_PCL_OPENMP
    else if (_method_type == MethodType::PCL_OPENMP) {
        omp_ndt.align(*output_cloud, init_guess);
        fitness_score = omp_ndt.getFitnessScore();
        t_localizer = omp_ndt.getFinalTransformation();
        has_converged = omp_ndt.hasConverged();
        final_num_iteration = omp_ndt.getFinalNumIteration();
    }
#endif

    //估计出的位姿
    //laser位姿->body位姿
    t_base_link = t_localizer * tf_ltob;

    //点云加入地图
    //通过laser位姿
    pcl::PointCloud<pcl::PointXYZ> cloudTemp;
    pcl::transformPointCloud(*scan_ptr, *transformed_scan_ptr, t_base_link);

    tf::Matrix3x3 mat_l, mat_b;

    mat_l.setValue(static_cast<double>(t_localizer(0, 0)), static_cast<double>(t_localizer(0, 1)),
                   static_cast<double>(t_localizer(0, 2)), static_cast<double>(t_localizer(1, 0)),
                   static_cast<double>(t_localizer(1, 1)), static_cast<double>(t_localizer(1, 2)),
                   static_cast<double>(t_localizer(2, 0)), static_cast<double>(t_localizer(2, 1)),
                   static_cast<double>(t_localizer(2, 2)));

    mat_b.setValue(static_cast<double>(t_base_link(0, 0)), static_cast<double>(t_base_link(0, 1)),
                   static_cast<double>(t_base_link(0, 2)), static_cast<double>(t_base_link(1, 0)),
                   static_cast<double>(t_base_link(1, 1)), static_cast<double>(t_base_link(1, 2)),
                   static_cast<double>(t_base_link(2, 0)), static_cast<double>(t_base_link(2, 1)),
                   static_cast<double>(t_base_link(2, 2)));

    // Update localizer_pose.
    localizer_pose.x = t_localizer(0, 3);
    localizer_pose.y = t_localizer(1, 3);
    localizer_pose.z = t_localizer(2, 3);
    mat_l.getRPY(localizer_pose.roll, localizer_pose.pitch, localizer_pose.yaw, 1);

    // Update ndt_pose.
    ndt_pose.x = t_base_link(0, 3);
    ndt_pose.y = t_base_link(1, 3);
    ndt_pose.z = t_base_link(2, 3);
    mat_b.getRPY(ndt_pose.roll, ndt_pose.pitch, ndt_pose.yaw, 1);

    //当前位置　map系
    current_pose.x = ndt_pose.x;
    current_pose.y = ndt_pose.y;
    current_pose.z = ndt_pose.z;
    current_pose.roll = ndt_pose.roll;
    current_pose.pitch = ndt_pose.pitch;
    current_pose.yaw = ndt_pose.yaw;

    T[3] = current_pose.x - previous_pose.x;
    T[5] = current_pose.y - previous_pose.y;
    T[4] = current_pose.z - previous_pose.z;
    T[0] = current_pose.roll - previous_pose.roll;
    T[2] = current_pose.pitch - previous_pose.pitch;
    T[1] = current_pose.yaw - previous_pose.yaw;

    std::cout << "output tf t" << T[3] << " " << T[5] << " " << T[4] << std::endl;
    std::cout << "output tf R" << T[0] << " " << T[2] << " " << T[1] << std::endl;

    for (pcl::PointCloud<pcl::PointXYZI>::const_iterator item = transformed_scan_ptr->begin();
         item != transformed_scan_ptr->end(); item++) {

        pcl::PointXYZ pNew;
        pNew.x = (double) item->x;
        pNew.y = (double) item->y;
        pNew.z = (double) item->z;

        cloudTemp.push_back(pNew);
    }
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "Sequence number: " << cloudIn->header.seq << std::endl;
    std::cout << "Number of scan points: " << scan_ptr->size() << " points." << std::endl;
    std::cout << "Number of filtered scan points: " << filtered_scan_ptr->size() << " points." << std::endl;
    std::cout << "transformed_scan_ptr: " << transformed_scan_ptr->points.size() << " points." << std::endl;
    std::cout << "map: " << map.points.size() << " points." << std::endl;
    std::cout << "NDT has converged: " << has_converged << std::endl;
    std::cout << "Fitness score: " << fitness_score << std::endl;
    std::cout << "Number of iteration: " << final_num_iteration << std::endl;
    std::cout << "(x,y,z,roll,pitch,yaw):" << std::endl;
    std::cout << "(" << current_pose.x << ", " << current_pose.y << ", " << current_pose.z << ", " << current_pose.roll
              << ", " << current_pose.pitch << ", " << current_pose.yaw << ")" << std::endl;
    std::cout << "Transformation Matrix:" << std::endl;
    std::cout << t_localizer << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;
    return cloudTemp;
}