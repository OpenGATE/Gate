

This CMakeList.txt allow sot build Root Geant4 and Gate with a single command.


- create a new folder (e.g. '<superbuild>/'). In the following change <superbuild> by this folder location.

- put CMakeList.txt in that folder

- from that folder, run 'ccmake .': choose option, type 'c' to configure, type 'g' to generate makefile

- compile by using 'make -j 6' (or replace 6 with the number of cores of your computer)

- setup the Geant4 environement variables (in order that is can find cross-sections and other data):
  source <superbuild>/install/bin/geant4.sh

- setup the Root, Geant4 and Gate environment variables :
   source <superbuild>/install/bin/thisroot.sh
   export LD_LIBRARY_PATH=<superbuild>/install/lib:$LD_LIBRARY_PATH
   export PATH=<superbuild>/install/bin:PATH

- then you can run Gate which is located into <superbuild>/install/bin/Gate

DETAILS:
- All sources are in <superbuild>/src
- All compiled source are in <superbuild>/src/XX-build
- binaries, libraries and data (cross-section) are installed in <superbuild>/install/
- G4 is build without QT support
