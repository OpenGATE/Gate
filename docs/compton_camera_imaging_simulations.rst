.. _compton_camera_imaging_simulations-label:

Compton camera imaging simulations: CCMod
=======================================

.. contents:: Table of Contents
   :depth: 15
   :local:

Introduction
------------

The Compton camera imaging system has been designed as an actor (see  :ref:`tools_to_interact_with_the_simulation_actors-label`) that collects the information of the hits in the different layers of the system. The following commands must be employed to add and attach the actor to a volume that contains the whole system::

	/gate/actor/addActor  ComptonCameraActor      [Actor Name]
	/gate/actor/[Actor Name]/attachTo             [Vol Name]            


The layers of the Compton camera work  as *Sensitive Detector* storing *Hits* (equivalent to volumes attached to crystalSD in PET/SPECT systems).
Therefore the digitizer modules described in :ref:`digitizer_and_readout_parameters-label` can be applied to get *Singles*.


A detailed description of CCMod can be found in the article `CCMod: a GATE module for Compton Camera imaging simulation (<https://doi.org/10.1088/1361-6560/ab6529>`


Defining the system 
-------------------
A Compton camera system is typically composed of two types of detectors: the scatterer and the absorber. These terms work as key words within the actor. The behavior of the  volumes associated to absorber and scatterer sensitive detectors (SD) is equivalent to the crystalSD  behavior  in  PET/SPECT systems. The sensitive detectors are specified with the following commands::

	/gate/actor/[Actor Name]/absorberSDVolume      [absorber Name]
	/gate/actor/[Actor Name]/scattererSDVolume     [scatterer Name]

For the absorber one single volume is foreseen whereas  multiple scatterer layers can be simulated.
At least one volume for the absorber and one volume for the scatterer are expected.
The total number of scatterer layers must also be specified using the following command::

	/gate/actor/[Actor Name]/numberOfTotScatterers [Num of scatterers]


When multiple scatterer layers are considered, if they are not created using a repeater (copies), the user must name them following a specific guideline. Once the name of one of the volumes is set, the rest of them must share the same name followed by a number to identify them. The  numbers are assigned in increasing order starting from 1. For example, 
if we have three scatterer layers  and we want to name them scatterer  the name of those volumes must be scatterer, scatterer1 and scatterer2.


There are no constrains for the geometry.

The output of the CC actor includes information about the energy and type of the source. When an ion source is employed, instead of the information of the ion, the information associated with one of the particles emitted in the  decays can be of interest. An extra option has been included in the actor  that allows to specify the parentID of the particle that is  going to be considered as a source. By default, this option is disabled::

	/gate/actor/[Actor Name]/specifysourceParentID 0/1

When it is set to 1, a text file must be included with a column of integers corresponding to the parentIDs  of the particles  that are going to be considered as primaries.

/gate/actor/[Actor Name]/parentIDFileName  [text file name]

For example,  in the case of  a 22Na source, we are interested in the 1274 keV emitted gamma-ray and the annihilation photons that can be identified using a value for the parentID of 2 and 4 respectively (at least using livermore or em opt4 physics list).
