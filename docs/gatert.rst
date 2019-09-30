.. _gatert-label:

GateRT
======

.. contents:: Table of Contents
   :depth: 15
   :local:

Some examples could be found here : https://davidsarrut.pages.in2p3.fr/gate-exercices-site/

**Disclaimer**:
*the tools dedicated to radiation therapy simulations provided in this GATE release are provided "as is" and on an "as available" basis without any representation or endorsement made and without warranty of any kind.*

**Location**:
The examples are located in the source code into to the folder : *Gate/examples/example_Radiotherapy/*. 


**Proton therapy examples**:

* Example 4 : Beam optics simulation in vacuum for a pencil beam + depth-dose profile in water. A root macro is provided to analysis the produced phase space files (PhS-Analysis.C).
* Example 5 : Treatment plan simulation of proton active scanning beam delivery (TPSPencilBeam source). A root macro is provided to analysis the produced phase space files (PhS-Analysis.C).
* Example 6 : Example of proton pencil beam in heterogeneous phantom (water, bones, Lung) with Pencil Beam Scanning source: comparison between dose to water and dose to dose to medium.  

**Carbon ion therapy examples**:

* Example 1 : Example of Carbon beam in water tank or in patient CT image.  Output is a 3D dose distribution map (with associated statistical uncertainty) and map of produced C11.

**Photon/electron therapy examples**:

* Example 2 : Example of photon beam in patient CT image.  Output is a 3D dose distribution map (with associated uncertainty). Two different navigators are tested NestedParameterized and Regionalized, with two number of materials). (This example is similar to the example1 presented on the wiki web site of Gate V6.1)
* Example 3 : Example of photon beam in patient CT image with IMRT irradiation. 100 slices with different MLC positions. (This example is similar to the example2 presented on the wiki web site of Gate V6.1)
* Example 7 : Example to use repeater/mover and both at the same time. (This example is similar to the example5 presented on the wiki web site of Gate V6.1)
* Example 8 : Photon beam from a Linac into a box with water/alu/lung. See Figure4 from [Jan et al PMB 2011].
* Example 9 : Electron beam from a Linac into a box with water/alu/lung. See Figure5 from [Jan et al PMB 2011].

**Radiography examples**:

* Example 10 : Radiography of a thorax phantom. Outputs are 3D dose distribution maps computed with the classical method and the accelerated (TLE) method.
