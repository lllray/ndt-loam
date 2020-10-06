^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package ndt_fuser
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.0.31 (2017-06-11)
-------------------
* testing tutorials on core packages. should be fixed now. 
* getting transform from tf tree in NDT_fuser_HMT
* Added generic deps.
* Started to move generic functions, such as io, conversions to a ndt_generic pkg.
* Initialize all fix, updated offline fuser with more params.
* Added soft constraint based registration.
* swtiched to eigen3 + remove dep in ndt_fuser
* added mpr_launch file
* Minor tweaks for 2d regi.
* Contributors: Henrik Andreasson, Malcolm Mielle, Todor Stoyanov

1.0.30 (2015-10-09)
-------------------
* added opengl explicit reference
* Contributors: Todor Stoyanov

1.0.29 (2015-10-08)
-------------------
*major code refactoring
* Contributors: Todor Stoyanov

1.0.28 (2014-12-05)
-------------------

1.0.27 (2014-12-05)
-------------------

1.0.26 (2014-12-03)
-------------------
* small fix to package.xmls and an update of visualization
* Contributors: Todor Stoyanov

1.0.25 (2014-12-01)
-------------------
* fixes for imported packages
* Contributors: Todor Stoyanov

1.0.24 (2014-11-25)
-------------------

1.0.23 (2014-11-25)
-------------------

1.0.20 (2014-11-21)
-------------------
* removing mrpt dependency
* map messages are now sent
* refactored fuser
* Contributors: Henrik Andreasson, Martin Magnusson, Todor Stoyanov, Tomasz Kucner

1.0.18 (2014-04-09)
-------------------

1.0.15 (2014-01-09)
-------------------
* fixes to makefiles wrt pcl
* Contributors: Todor Stoyanov

1.0.13 (2014-01-09)
-------------------

1.0.12 (2013-12-03)
-------------------
* Added new install targets for launch files
* Contributors: Todor Stoyanov

1.0.10 (2013-12-03)
-------------------

1.0.9 (2013-12-02)
------------------
* removed deprecated dependancy on mrpt-graphslam
* added the empty directory to hold maps in hydro
* hydro release
* Contributors: Todor Stoyanov, Tomasz Kuncer

1.0.8 (2013-12-02)
------------------
* "1.0.8"
* changelogs updated
* Removed legacy dependency to mrpt-graphslam
* added the empty directory for maps in groovy
* Contributors: Todor Stoyanov

1.0.7 (2013-11-28)
------------------
* "1.0.7"
* changelogs update
* Added release flags to all CMake files
* Re-organization of include files to follow ros convention, lots of changes
* Contributors: Todor Stoyanov

1.0.6 (2013-11-27 20:13)
------------------------
* "1.0.6"
* slight progress, compilation errors now
* Contributors: Todor Stoyanov

1.0.5 (2013-11-27 19:52)
------------------------
* "1.0.5"
* Contributors: Todor Stoyanov

1.0.4 (2013-11-27 19:40)
------------------------
* "1.0.4"
* Contributors: Todor Stoyanov

1.0.3 (2013-11-27 19:26)
------------------------
* "1.0.3"
* prepairing for second release candidate
* CMake files fixed to output in the correct place
* Contributors: Todor Stoyanov

1.0.2 (2013-11-27 13:58)
------------------------
* "1.0.2"
* Contributors: Todor Stoyanov

1.0.1 (2013-11-27 12:33)
------------------------
* "1.0.1"
* added changelog files to stream
* removed message gen that was not needed and generating scary warnings
* removed the precompiled binaries from the branch. those should not go on the repo
* the removal of rosbuild remains
* compiled packages ndt_fuser  ndt_map  ndt_map_builder  ndt_mcl  ndt_registration  ndt_visualisation  perception_oru  pointcloud_vrml
* Contributors: Todor Stoyanov, Tomasz Kuncer
