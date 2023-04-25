#Ubuntu 22.04
#user: vgate
#psswd: virtual

sudo apt update
sudo apt install build-essential
sudo apt-get install -y git \
                        cmake \
                        cmake-curses-gui \
                        freeglut3-dev \
                        libglew-dev \
                        libglm-dev \
                        libqt5x11extras5-dev \
                        qttools5-dev \
                        libxpm-dev \
                        libxft-dev \
                        libxmu-dev \
                        git-lfs \
                        libssl-dev \
                        libxml2-dev \
                        fftw3-dev
                     
cd
mkdir Software
cd Software
mkdir VTK ITK RTK vv root Geant4 Gate
cd VTK
mkdir src bin install
git clone -b v9.0.3 https://github.com/Kitware/VTK.git src
cd bin
cmake ../src -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/vgate/Software/VTK/install -DVTK_MODULE_ENABLE_VTK_GUISupportQt=YES -DVTK_MODULE_ENABLE_VTK_RenderingQt=YES -DVTK_MODULE_ENABLE_VTK_ViewsQt=YES
make install
cd ..
rm -rf bin src

cd
cd Software/ITK
mkdir src bin install
git clone -b v5.2.1 https://github.com/InsightSoftwareConsortium/ITK.git src
cd bin
ccmake ../src -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/home/vgate/Software/ITK/install -DModule_ITKVtkGlue=ON -DVTK_DIR=/home/vgate/Software/VTK/install/lib/cmake/vtk-9.0/
make install
cd ..
rm -rf bin src

cd
cd Software/RTK
mkdir src bin
git clone -b v5.3.0 https://github.com/InsightSoftwareConsortium/ITK.git src
cd bin
ccmake ../src -DBUILD_TESTING=OFF -DModule_RTK=ON -DITK_USE_FFTWD=ON -DITK_USE_FFTWF=ON -DITK_USE_SYSTEM_FFTW=ON
make

cd
cd Software/vv
mkdir src bin install
git clone https://github.com/open-vv/vv.git src
cd bin
ccmake ../src -DITK_DIR=/home/vgate/Software/ITK/install/lib/cmake/ITK-5.2/ -DCMAKE_INSTALL_PREFIX=/home/vgate/Software/vv/install
make install
cd ..
rm -rf src bin

cd
cd Software/root
mkdir src bin
git clone -b v6-26-08 https://github.com/root-project/root.git src
cd bin
ccmake ../src -Dpython=OFF
make

cd
cd Software/Geant4
mkdir src bin install data
git clone -b v11.1.1 https://github.com/Geant4/geant4.git src
cd bin
ccmake ../src -DGEANT4_INSTALL_DATA=ON -DGEANT4_BUILD_MULTITHREADED=OFF -DGEANT4_INSTALL_DATADIR=/home/vgate/Software/Geant4/data -DCMAKE_INSTALL_PREFIX=/home/vgate/Software/Geant4/install -DGEANT4_BUILD_MULTITHREADED=OFF -DGEANT4_USE_QT=ON -DGEANT4_USE_OPENGL_X11=ON
make install
cd ..
rm -rf src bin

cd
cd Software
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.7.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.7.0+cpu.zip

cd
cd Software
wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip
unzip fiji-linux64.zip

cd
cd Software
git clone https://github.com/OpenGATE/GateContrib.git

echo 'export PATH=/home/vgate/Software/RTK/bin/bin:${PATH}' >> /home/vgate/.bashrc
echo 'export PATH=/home/vgate/Software/vv/install/bin:${PATH}' >> /home/vgate/.bashrc
echo 'export PATH=/home/vgate/Software/Fiji.app:${PATH}' >> /home/vgate/.bashrc
echo 'export ROOTSYS=/home/vgate/Software/root/bin/bin/root-config' >> /home/vgate/.bashrc
echo 'export PATH=${ROOTSYS}:${PATH}' >> /home/vgate/.bashrc
echo 'source /home/vgate/Software/root/bin/bin/thisroot.sh' >> /home/vgate/.bashrc
echo 'source /home/vgate/Software/Geant4/install/bin/geant4.sh' >> /home/vgate/.bashrc

cd
cd Software/Gate
mkdir src bin
git clone -b v9.3 https://github.com/OpenGATE/Gate.git src
cd bin
ccmake ../src -DGATE_USE_RTK=ON -DGATE_USE_TORCH=ON -DTorch_DIR=/home/vgate/Software/libtorch/share/cmake/Torch -DGATE_COMPILE_GATEDIGIT=ON
make
echo 'export PATH=/home/vgate/Software/Gate/bin:${PATH}' >> /home/vgate/.bashrc

echo 'export LD_LIBRARY_PATH=/home/vgate/Software/ITK/install/lib:${LD_LIBRARY_PATH}' >> /home/vgate/.bashrc
echo 'export LD_LIBRARY_PATH=/home/vgate/Software/VTK/install/lib:${LD_LIBRARY_PATH}' >> /home/vgate/.bashrc



