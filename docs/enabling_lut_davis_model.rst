.. _enabling_lut_davis_model-label:

Enabling LUT Davis Model
========================

.. contents:: Table of Contents
   :depth: 15
   :local:

Manually Modifying Geant4
-------------------------

(Necessary before Geant4 release Summer 2017) 

In order to use the Davis LUT model Geant4 has to be extended. This documentation provides detailed step-by-step compiling instructions.  
 
1) Follow compiling instructions here: :ref:`geant4-label` stop before running ccmake, proceed with step 2.
2) Get the modified code and Look-up-Tables: 5 `Geant4 classes <https://github.com/OpenGATE/GateContrib/tree/master/misc/ModifiedG4Classes>`_ to be replaced in Geant4 10.2 or 10.3, respectively and 18 `LUTs <https://midas3.kitware.com/midas/download/item/321835/Davis_LUTs.tar.gz>`_ provided as a tar.gz archive 
3) Replace header files in your local Geant4 directory 
    - Go to /PATH_TO_yourGEANT4Directory/source/materials/include 
        - Replace the "G4SurfaceProperty.hh" file with the provided file 
        - Replace the "G4OpticalSurface.hh" file with the provided file 
    - Go to /PATH_TO_yourGEANT4Directory/source/materials/src 
        - Replace the "G4OpticalSurface.cc" file with the provided file 
    - Go to /PATH_TO_yourGEANT4Directory/source/processes/optical/include 
        - Replace the "G4OpBoundaryProcess.hh" file with the provided file 
    - Go to /PATH_TO_yourGEANT4Directory/source/processes/optical/src 
        - Replace the "G4OpBoundaryProcess.cc" file with the provided file 
4) Open "G4OpticalSurface.hh" in /PATH_TO_yourGEANT4Directory/source/materials/include   
    - In line 270 set the path of the Davis_LUTs-folder that was downloaded in step 2 
    - *G4String PathToLUT="/home/…/Davis_LUTs";*
5) Go on with ccmake process in of step 1. and finish as documented.
6) To enable modifications of Geant4 in GATE: Follow compiling instructions: :ref:`gate-label`
7) In the ccmake process set CMake options GATE_USE_OPTICAL and GATE_USE_DAVIS to ON. (Run provided example to validate installation for LUT Davis model and compare to provided output file.)
