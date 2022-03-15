:orphan:

.. _package_requirements-label:

Installation of required dependendencies on some specific platforms
===================================================================


*Note: some of this information may be out of date for the 8.1 release.*

 This list may change frequently, last update: Dec. 31 2015

Ubuntu 18.04.2 LTS (for GATE v8.2 w/ Geant4 10.5 p01)
-----------------------------------------------------

In Terminal, **type** ::

   sudo apt-get update
   sudo apt-get install <package_1 here> <package_2 here> ... <package_N here>

to install the packages. Replace <package_X here> with the correct packages. For example::

   sudo apt-get install cmake cmake-curses-gui build-essential libqt4-opengl-dev qt4-qmake libqt4-dev libx11-dev libxmu-dev libxpm-dev libxft-dev

**The following packages are required for GATE v8.2 with minimal options turned ON (Qt5, OpenGL turned on)** ::

  cmake-curses-gui libqt5-default libxmu-dev

Note: In the event a package does not exist in the Ubuntu repository, you can search for a potential replacement by typing ::

   sudo apt-cache search <search_term>

Ubuntu 16.04.2 LTS (for GATE v8.0 w/ Geant4 10.3 p01) and Ubuntu 16.04 LTS (for GATE v7.2 w/ Geant4 10.2 p01)
-------------------------------------------------------------------------------------------------------------

In Terminal, **type** ::

   sudo apt-get update
   sudo apt-get install <package_1 here> <package_2 here> ... <package_N here>

to install the packages. Replace <package_X here> with the correct packages. For example::

   sudo apt-get install cmake cmake-curses-gui build-essential libqt4-opengl-dev qt4-qmake libqt4-dev libx11-dev libxmu-dev libxpm-dev libxft-dev

**The following packages are required for GATE v7.2/8.0 with minimal options turned ON (For example, GATE_USE_GPU = OFF)** ::

  cmake cmake-curses-gui build-essential libqt4-opengl-dev qt4-qmake libqt4-dev libx11-dev libxmu-dev libxpm-dev libxft-dev

**The following package is required for GATE v8.0 with GATE_USE_OPTICAL turned ON** ::

  libxml2-dev

Note: In the event a package does not exist in the Ubuntu repository, you can search for a potential replacement by typing ::

   sudo apt-cache search <search_term>

Ubuntu 14.04.3 LTS (for GATE v7.1 w/ Geant4 10.1 p02) and Ubuntu 12.04.5 LTS (for GATE v7.0 w/ Geant4 9.6 p04)
--------------------------------------------------------------------------------------------------------------

In Terminal, type ::

   sudo apt-get update
   sudo apt-get install <package_1 here> <package_2 here> ... <package_N here>

to install the packages. Replace <package_X here> with the correct packages. For example::

   sudo apt-get install cmake cmake-curses-gui build-essential libqt4-opengl libqt4-opengl-dev libqt4-core qt4-qmake libqt4-dev libX11-dev libxmu-dev

**The following packages are required for GATE v7.0/7.1 with minimal options turned ON (For example, GATE_USE_GPU = OFF)** ::

  cmake cmake-curses-gui build-essential libqt4-opengl libqt4-opengl-dev libqt4-core qt4-qmake libqt4-dev libX11-dev libxmu-dev

Note: In the event a package does not exist in the Ubuntu repository, you can search for a potential replacement by typing::

   sudo apt-cache search <search_term>

**WARNING: For GATE validation, please refer to** http://wiki.opengatecollaboration.org/index.php/Validating_Installation

**GATE benchmark results here** http://www.opengatecollaboration.org/PETBenchmark **appears to be outdated.**



Fedora
------

Use::

   sudo yum install <packages here>

to install the packages. Replace <packages here> with the correct packages.

The following packages are required::

  freeglut freeglut-devel gtkglext-devel gtkglext-libs gcc gcc-gfortran gcc-c++ compat-libgfortran-41
  libgfortran glibc-kernheaders glibc-headers glibc-devel glibc glibc-static openmotif openmotif-devel
  libXaw-devel libXaw libXpm-devel libXpm libxml2-devel libxml2 xerces-c-devel qt qt-devel qt-x11 binutils
  libX11-devel libXft-devel libXext-devel ncurses-devel pcre-devel mesa-libGL-devel mesa-libGL gtkglarea2
  gtkglarea2-devel InventorXt InventorXt-devel lesstif lesstif-devel libfftw3-3 libfftw3-dev libfftw3-doc

Scientific Linux 6
------------------

Use::

   sudo yum install <packages here>

to install the packages. Replace <packages here> with the correct packages.

The following packages are required::

  freeglut freeglut-devel gtkglext-devel gtkglext-libs gcc gcc-gfortran gcc-c++ compat-libgfortran-41
  libgfortran glibc-kernheaders compat-glibc-headers glibc-headers glibc-devel glibc glibc-static
  compat-libstdc++ compat-glibc openmotif openmotif-devel libXaw-devel libXaw libXpm-devel libXpm libxml2-devel
  libxml2 xerces-c-devel qt qt-devel qt-x11 binutils libX11-devel libXft-devel libXext-devel ncurses-devel
  pcre-devel mesa-libGL-devel mesa-libGL libxml2-dev libxml2 gtkglarea2 gtkglarea2-devel fftw-devel fftw2-devel
  fftw fftw2
