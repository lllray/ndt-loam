#include <ndt_visualisation/ndt_viz.h>
#include <ndt_costmap/costmap.h>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;
using namespace perception_oru;
  
int main(int argc, char **argv){

    std::string base_name;
    double resolution;
    double robot_height;
    //double 

    po::options_description desc("Allowed options");
    desc.add_options()
	("help", "produce help message")
	("file-name", po::value<string>(&base_name), "location of the ndt map you want to view")
	("resolution", po::value<double>(&resolution)->default_value(1.), "resolution of the map (necessary due to bug in loading, should be fixed soon)")
    ("robot_height", po::value<double>(&robot_height)->default_value(1.86), "robvot height")
	;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (!vm.count("file-name") || !vm.count("resolution"))
    {
	cout << "Missing arguments.\n";
	cout << desc << "\n";
	return 1;
    }
    if (vm.count("help"))
    {
	cout << desc << "\n";
	return 1;
    }

    std::cout<<"loading "<<base_name<<" at resolution "<<resolution<<std::endl;

    //load NDT map. Only Lazy grid supported!
    perception_oru::NDTMap ndmap(new perception_oru::LazyGrid(resolution));
    ndmap.loadFromJFF(base_name.c_str());
    
    NDTCostmap cmap(&ndmap);
    cmap.processMap(robot_height, -1);
    cmap.saveCostMapIncr("cmap",0.25);
    
    //create visualizer
    NDTViz *viewer = new NDTViz(true);
    //display map
    viewer->plotNDTSAccordingToClass(-1,&ndmap);
    char c;
    std::cin >> c;
    delete viewer;

}
