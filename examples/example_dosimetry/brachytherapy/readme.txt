

* Authors:
 - Lydia Maigne
 - Yann Perrot
 - David Sarrut

* Step 1: generate the phase space

Run main.mac to obtain a ROOT phase space file TitaniumOuter.root in the output/ repository
- StandardPhys.mac describes the physics processes
- sourceIodine.mac describes the energy spectrum of 125I source

In the folder data/
- ct-2mm-HU2mat.txt describes the Hounsfield range values allocated to tissues
- ct-2mm.mhd is the file describing the CT scan

* Step 2:

First, copy or move the phase space file (TitaniumOuter.root) into data/ folder

Run mainReadPhs.mac to obtain the dose distribution dose-dose.raw in the output/ repository
- phaseSpace79.mac describes the virtual seeds attached to the ROOT phase space file
- virtualVolume79.mac creates the 79 virtual tiny volumes and their location in the CT scan

Warning, it requires about 3GB RAM (because the phase space is
reloaded for each 80 sources).

In the repository output/ you can find:
- output-Dose.mhd is the absolute dose distribution map
- output-Dose-Squared.mhd is the squared dose map
- output-Dose-Uncertainty.mhd is dose uncertainty map

* Visualisation
You can use vv : vv data/ct-2mm.mhd --fusion output/output-Edep.mhd
