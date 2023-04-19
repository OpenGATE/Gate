.. _compton_camera_imaging_simulations-label:

Compton camera imaging simulations: CCMod (TO BE ADDED SOON IN GATE 9.3)
=======================================

.. contents:: Table of Contents
   :depth: 15
   :local:

Introduction
------------

The Compton camera imaging system has been designed as an actor (see  :ref:`tools_to_interact_with_the_simulation_actors-label`) that collects the information of the *Hits* in the different layers of the system. The following commands must be employed to add and attach the actor to a volume that contains the whole system::

	/gate/actor/addActor  ComptonCameraActor      [Actor Name]
	/gate/actor/[Actor Name]/attachTo             [Vol Name]            


The layers of the Compton camera work  as *Sensitive Detectors* storing *Hits* (equivalent to volumes attached to crystalSD in PET/SPECT systems).
Therefore the digitizer modules described in :ref:`digitizer_and_readout_parameters-label` can be applied to *Hits* get *Singles*.


A detailed description of CCMod can be found in the article `CCMod: a GATE module for Compton Camera imaging simulation <https://doi.org/10.1088/1361-6560/ab6529>`


Defining the system 
-------------------
A Compton camera system is typically composed of two types of detectors: the scatterer and the absorber. These terms work as key words within the actor. The behavior of the  volumes associated to absorber and scatterer *Sensitive detectors* is equivalent to the crystalSD  behavior  in  PET/SPECT systems. The sensitive detectors are specified with the following commands::

	/gate/actor/[Actor Name]/absorberSDVolume      [absorber Name]
	/gate/actor/[Actor Name]/scattererSDVolume     [scatterer Name]

For the absorber one single volume is foreseen whereas  multiple scatterer layers can be simulated.
At least one volume for the absorber and one volume for the scatterer are expected.
The total number of scatterer layers must also be specified using the following command::

	/gate/actor/[Actor Name]/numberOfTotScatterers [Num of scatterers]


When multiple scatterer layers are considered, if they are not created using a repeater (copies), the user must name them following a specific guideline. Once the name of one of the volumes is set, the rest of them must share the same name followed by a number to identify them. The  numbers are assigned in increasing order starting from 1. For example, 
if we have three scatterer layers  and we want to name them scatterer  the name of those volumes must be scatterer, scatterer1 and scatterer2.


There are no constrains for the geometry.



Digitization 
-------------

The main  purpose of the digitization is to simulate the behavior of the detector response. The same data structures (i. e. *Hits*, *Singles*, *Coincidences*) as in PET/SPECT systems have been employed to be able to share the digitizer modules between the systems and the CCMod. Therefore, the digitizer modules described in :ref:`digitizer_and_readout_parameters-label` can be  directly applied to the Compton camera by inserting the modules using the following command. The key word *layers* instead of *singles* must be employed::

	/gate/digitizer/layers/insert [Module name]

Most of the modules available for systems are global modules; thus, they are applied to all the considered sensitive volumes. However, a Compton camera system is typically composed of two different types of detectors (the scatterer and the absorber). Therefore, it is useful to apply a different digitization chain to each of them. To this end, in addition to the global modules, several local modules have been developed that are applied to specific volumes using the following command::

	/gate/digitizer/layers/[Module name]/chooseNewVolume [SD volume name]



List of additional digitizer modules
-------------------------------------
Here, there is a list of the additional developed modules.
..
	grid discretization (local module), clustering (local and global modules), ideal adder (local and global modules), DoI modeling (global module), time delay (local module), 3D spatial resolution (local module), multiple single rejection (local module), energy threshold module with different policies for effective energies (local and global modules).



Grid discretization  module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module allows to simulate the  readout of strip and pixelated detectors. Since it is a local module, the first thing is to attach it to a specific volume that must be acting as a SD::

	/gate/digitizer/layers/insert gridDiscretization
	/gate/digitizer/layers/gridDiscretization/chooseNewVolume [volName]

The number of the strips/pixels must be specified in X and Y directions. In addition, the width of the strips/pixel and an offset can be specified to take into account the insensitive material in the detector layer::

	/gate/digitizer/layers/gridDiscretization/[volName]/setNumberStripsX [Nx]
	/gate/digitizer/layers/gridDiscretization/[volName]/setNumberStripsY [Ny]
	/gate/digitizer/layers/gridDiscretization/[volName]/setStripOffsetX   [offSet_x]
	/gate/digitizer/layers/gridDiscretization/[volName]/setStripOffsetY [offSet_y]
	/gate/digitizer/layers/gridDiscretization/[volName]/setStripWidthX [size_x]
	/gate/digitizer/layers/gridDiscretization/[volName]/setStripWidthY [size_y]

The *hits* detected in the strips/pixels are merged at the center of the strip/pixel in each spatial direction. When strips are defined in both spatial directions, only the hits in the volume defined by the intersection of two strips are stored; thus, generating pixels.

When the grid discretization module is employed to reproduce the response of strip detectors, it should be generally applied followed by a strip activation energy threshold and a multiple single rejection module to avoid ambiguous strip-intersection identification.  

On the other hand, when pixelated crystals are simulated, it can be of interest to  apply the readout at the level of blocks composed of several pixels. The number of readout blocks can be set individually in each direction using the following commands::

	/gate/digitizer/layers/gridDiscretization/[volName]/setNumberReadOutBlocksX  [NBx]
	/gate/digitizer/layers/gridDiscretization/[volName]/setNumberReadOutBlocksY  [NBy]

The energy in the block corresponds to the sum of the deposited energy and the position to the  energy weighted centroid position in the pixels that composed the block.

Clustering module
~~~~~~~~~~~~~~~~~
This module has been designed with monolithic crystals read-out by segmented photodetectors in mind. Both versions the global module and its local counterpart have been developed::

	/gate/digitizer/layers/insert clustering

or for the local counterpart::

	/gate/digitizer/layers/insert localClustering
	/gate/digitizer/layers/localClustering/chooseNewVolume [volName]

The hits located within the same volume are regrouped by distance, creating clusters. If a detected *hit* is closer than a specified accepted distance to one of the clusters, it is added to the closest one; otherwise, it generates a new cluster. The *hits* are added summing their deposited energies and computing the energy-weighted centroid position. If two clusters are closer than the accepted distance they are merged following the same criteria. If requested, events with multiple clusters in the same volume can be rejected::

	/gate/digitizer/layers/clustering/setAcceptedDistance [distance plus units]
	/gate/digitizer/layers/clustering/setRejectionMultipleClusters [0/1]

or for the local counterpart::

	/gate/digitizer/layers/localClustering/setAcceptedDistance [distance plus units]
	/gate/digitizer/layers/localClustering/setRejectionMultipleClusters [0/1]


Ideal adder module
~~~~~~~~~~~~~~~~~~~
This module has been designed with the aim of recovering the exact Compton kinematics to enable further studies.

The adderCompton module was designed with the same aim.  However, it does not work properly when there are several photonic hits with secondary electronic hit associated in the same volume since the module only distinguish between photonic and electronic hits. The adderCompton module is designed so that the energy of the electronic *hits* is added to the last photonic hit in the same  volume. Therefore, when there are two photonic hits in the same volume, the energy of all the electronic hits is added to the second photonic hit  leaving the  first hit  in general with an incorrect  null energy deposition associated.

In order to develop an adder that  allows us to recover the exact Compton kinematics also when several primary photonic hits occur in the same volume, extra information such as post-step process, creator process, initial energy of the track, final energy, trackID and parentID was  added to the pulses. This module creates a *single* from each primary photon *hit* that undergoes a Compton, Photoelectric or Pair Creation interaction. Additional information, such as the energy of the photon that generates the pulse before (*energyIni*) and after (*energyFinal*) the primary interaction is included to be able to recover the ideal Compton kinematics, hence its name. These attributes have invalid values (-1) when this module is not applied. The deposited energy value (*energy*) of each pulse should correspond to the sum of the deposited energy of the primary hit and all the secondary hits produced by it. The deposited energy was validated using livermore physics list. Note that the method applied to obtained  the deposited energy (*energy attribute) is not robust and may lead to incorrect values for other physics list.
 
Both versions the global module and its local counterpart have been developed.  They can be employed using the following command::

	/gate/digitizer/layers/insert adderComptPhotIdeal

or::

	/gate/digitizer/layers/insert adderComptPhotIdealLocal
	/gate/digitizer/layers/adderComptPhotIdealLocal/chooseNewVolume [volName]

 
The option to reject those events in which the primary photon undergoes at least one interaction different from Compton or Photoelectric  is included  in the global module using the following command:::

	/gate/digitizer/layers/insert/rejectEvtOtherProcesses [1/0]

In order to get one *single* per volume, the user can apply another module afterwards such as the standard adder to handle multiple interactions.


Energy thresholder module
~~~~~~~~~~~~~~~~~~~~~~~~~
This module apply an energy threshold for the acceptance of pulses. By default, the threshold is applied to the deposited energy. Both versions the global module and its local counterpart have been developed. They can be added using the following commands.::

	/gate/digitizer/layers/insert energyThresholder
	/gate/digitizer/layers/energyThresholder/[volName]/setThreshold [energy]

or::

	/digitizer/layers/insert localEnergyThresholder
	/gate/gate/digitizer/layers/localEnergyThresholder/chooseNewVolume [volName]
	/gate/digitizer/layers/localEnergyThresholder/[volName]/setThreshold [energy]

This threshold is applied to an effective energy that can be obtained using different criteria. Two options have been implemented namely deposited energy and solid angle weighted energy.  In order to explicitly specify that the threshold is applied to the deposited energy, the following command should be employed:::

	/gate/digitizer/layers/energyThresholder/setLaw/depositedEnergy

or::

	/gate/digitizer/layers/localEnergyThresholder/[volName]/setLaw/depositedEnergy


For the solid angle weighted energy policy, the effective energy for each pulse is calculated multiplying the deposited energy by a factor that represents the fraction of the solid angle from the pulse position subtended by a virtual pixel centered in the X-Y pulse position at the detector layer readout surface. To this end, the size of the pixel and detector readout surface must be specified. Those characteristics are included using the following commands::


	/gate/digitizer/layers/energyThresholder/setLaw/solidAngleWeighted
	/gate/digitizer/layers/energyThresholder/solidAngleWeighted/setRentangleLengthX [szX]
	/gate/digitizer/layers/energyThresholder/solidAngleWeighted/setRentangleLengthY [szY]
	/gate/digitizer/layers/energyThresholder/solidAngleWeighted/setZSense4Readout [1/-1]

or for the local counterpart::

	/gate/digitizer/layers/localEnergyThresholder/[volName]/setLaw/solidAngleWeighted
	/gate/digitizer/layers/localEnergyThresholder/[volName]/solidAngleWeighted/setRentangleLengthX [szX]
	/gate/digitizer/layers/localEnergyThresholder/[volName]/solidAngleWeighted/setRentangleLengthY [szY]
	/gate/digitizer/layers/localEnergyThresholder/[volName]/solidAngleWeighted/setZSense4Readout [1/-1]


If at least the effective energy of one of the pulses is over the threshold, all the pulses  corresponding to the same event registered in the studied sensitive volume are stored, otherwise they are rejected.


The global energy thresholder with the default option (deposited energy law) is  equivalent to the already available  global thresholder. 


DoI modeling
~~~~~~~~~~~~

The DoI modeling digitizer is applied using the following command.::

	/gate/digitizer/layers/insert DoImodel
..
	 It is a global module. The local counterpart can be useful::



The different considered DoI models can be applied to two readout geometries (Schaart et al. 2009): front surface (entrance surface) readout, in which the photodetector is placed on the crystal surface facing the radiation source, and conventional back-surface (exit surface) readout. To this end, the  growth-direction of the DoI must be specified using the command.::

	/gate/digitizer/layers/DoImodel/setAxis [0 0 1]

In the above example the growth-direction of the DoI is set to  the growth direction of the Z-axis.
The criterion for the DoI growth is set towards the readout surface and thereby the DoI value in that surface corresponds to the thickness of the crystal. The opposite surface of the readout surface is referred to as exterior surface. Therefore, the  different uncertainty models implemented can be applied to the different readout configurations.

Two options are available for the DoI modelling: dual layer structure and exponential function for the DoI uncertainty. The dual layer model discretizes the ground-truth DoI into  two positions in the crystal. If the position of the pulse is recorded in the half of the crystal closer to the readout surface, the DoI is set to the central section, otherwise it is set to the exterior surface.
This model can be selected using the following command::

	/gate/digitizer/layers/DoImodel/setDoIModel dualLayer

The DoI exponential uncertainty is modeled as a negative exponential function in the DoI growth-direction. FWHM value at the exterior surface (maximum uncertainty) and the exponential decay constant must be set as input parameters. This uncertainty model and the necessary parameters can be  loaded using the following commands.::


	/gate/digitizer/layers/DoImodel/setDoIModel DoIBlurrNegExp
	/gate/digitizer/layers/DoImodel/DoIBlurrNegExp/setExpInvDecayConst [length]
	/gate/digitizer/layers/DoImodel/DoIBlurrNegExp/setCrysEntranceFWHM [length]



Local Time delay module
~~~~~~~~~~~~~~~~~~~~~~~

This local module delays the time value of the detected pulses in a specified *Sensitive Detector* volume. It can be useful in a Compton camera system, for instance, to delay the *singles* in the scatterer detector when the absorber gives the coincidence trigger::

	/gate/digitizer/layers/insert localTimeDelay
	/gate/digitizer/layers/localTimeDelay/chooseNewVolume [volName]
	/gate/digitizer/layers/localTimeDelay/[volName]/setTimeDelay [time value]


Local time resolution
~~~~~~~~~~~~~~~~~~~~~
In addition to the global time resolution module described in section :ref:`digitizer_and_readout_parameters-label`  a  local version has been included in order to be able to set different time resolutions to the different layers::

	/gate/digitizer/layers/insert localTimeResolution
	/gate/digitizer/layers/localtimeResolution/setTimeResolution [FWHM value]

Local 3D  spatial resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This local module sets independently  a Gaussian spatial resolution in each spatial direction.
The module is inserted using the following command::

	/gate/digitizer/layers/insert sp3Dlocalblurring
	/gate/digitizer/layers/sp3Dlocalblurring/chooseNewVolume [vol name]

and the sigma of the Gaussian function in each direction is set::

	/gate/digitizer/layers/sp3Dlocalblurring/[vol name]/setSigma [vector (length)]



Local Multiple rejection module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a local module that allows you to discard multiple pulses. It can be inserted using the following commands.::

	/gate/digitizer/layers/insert localMultipleRejection
	/gate/digitizer/layers/localMultipleRejection/chooseNewVolume [vol]

The definition of  what is considered multiple pulses must be set. Two options are available: more than one pulse in the same volume name or more than one pulses in the same volumeID.
When several identical volumes are needed, for example for several scatterer layers, they are usually created as copies using a repeater. In that case, all volumes share the same name but they have different volumeID.  The difference between the rejection based on volume name and volumeID is important in those cases.
These options are selected using the following command line.::

	/gate/digitizer/layers/localMultipleRejection/[vol]/setMultipleDefinition [volumeID/volumeName]

Then, the rejection can be set to the whole event or only to those pulses within the same volume name or volumeID where the multiplicity happened.::

	/gate/digitizer/layers/localMultipleRejection/[vol]/setEventRejection [1/0]



Sorter
-------

The sorter developed in GATE for PET systems has been adapted for the CCMod, see :ref:`coincidence_sorter-label`. Same  command is employed.::

	/gate/digitizer/Coincidences/setWindow [time value]

An additional option has been included to allow only *singles* in the absorber layer to open its own time window, i. e.  absorber coincidence trigger. By default, this option is disabled.
In order to enable it the following command must be employed::

	/gate/digitizer/Coincidences/setTriggerOnlyByAbsorber 1

Different coincidence acceptance policies are available for Compton camera: *keepIfMultipleVolumeIDsInvolved*, *keepIfMultipleVolumeNamesInvolved*, *keepAll*.
They can be selected using the following command line::

	/gate/digitizer/Coincidences/setAcceptancePolicy4CC keepAll

*KeepAll* policy accepts all coincidences, no restriction applied.

*KeepIfMultipleVolumeIDsInvolved* policy accepts *coincidences* with at least two *singles* in different volumeIDs.  

*KeepIfMultipleVolumeNamesInvolved* is the default *coincidence* acceptance policy. *Coincidences* are accepted if at least two of the *singles*  within the *coincidence* are recorded in different SD  volume names. Volumes created by a repeater have same volume name but different volumeID.
 
Coincidence processing
-----------------------
The described modules in  :ref:`coincidence_processing-label` to process coincidences in PET systems such as dead-time or
memory buffer  can be in principle applied directly to CCMod using the same commands::

	/gate/digitizer/name sequenceCoincidence  
	/gate/digitizer/insert coincidenceChain
	/gate/digitizer/sequenceCoincidence/addInputName Coincidences

However, since they are designed for PET systems, some of them reject multiple *coincidences* (more than two *singles*).

Coincidence Sequence Reconstruction (CSR)  module has been included for CCMod. It is a *coincidence* processor which modifies the order of the *singles* within a *coincidence* to generate a *sequence coincidence*::

	/gate/digitizer/sequenceCoincidence/insert [name]

Different policies have been implemented to order the *singles* within a *coincidence*: randomly, by increasing single time-stamp value (ideal), axial distance to the source (first scatterer then absorber) or deposited energy. Those policies can be selected using the following commands.::


	/gate/digitizer/sequenceCoincidence/[name]/setSequencePolicy randomly
	/gate/digitizer/sequenceCoincidence/[name]/setSequencePolicy singlesTime
	/gate/digitizer/sequenceCoincidence/[name]/setSequencePolicy axialDist2Source
	/gate/digitizer/sequenceCoincidence/[name]/setSequencePolicy lowestEnergyFirst

In addition, a policy based on the so-called revan analyzer from Megalib (Zoglauer et al. 2008), known as Classic Coincidence Sequence Reconstruction (CCSR) has been included.
.. 
	(It is disabled from the messenger since the  the errors in energy and posiiton are not properly included in the pulses)

	/gate/digitizer/sequenceCoincidence/[name]/setSequencePolicy revanC_CSR



Data output
-----------
Output data is saved  using the following command::

	/gate/actor/[Actor Name]/save   [FileName]
Data can be saved in .npy, .root or .txt format. The format is taken from the extension included in the chosen FileName. 
The information of the *Hits*, *Singles*, *Coincidences* and Coincidence chains can be stored::

	/gate/actor/[Actor Name]saveHitsTree         [1/0]                  
	/gate/actor/[Actor Name]/saveSinglesTree       [1/0]                 
	/gate/actor/[Actor Name]/saveCoincidencestTree     [1/0]              
	/gate/actor/[Actor Name]/saveCoincidenceChainsTree  [1/0] 

For each data format (*Hits*, *Singles*, *Coincidences*,  processed coincidence name) a new file is generated with  the label of the data included.
For examples if the FileName is test.root, then *Singles* are saved in the file called test_singles.root.

Most of the  information  in the output file can be enabled or disabled by the user. 
For example, the information of the energy deposition can be disabled using the following command::


	/gate/actor/[Actor Name]/enableEnergy 0


An additional file with electron escape information can be stored::
	
	/gate/actor/CC_digi_BB/saveEventInfoTree            [1/0]

If this option is enabled and the chosen general FileName is for example *test.root*,  a new file *test_eventGlobalInfo.root* is generated.
For each electron that goes through a SD volume, a flag that indicates if the electron enters or exits the volume, the SD detector volume name, the energy of the electron, the eventID and the runID are stored.


Optional additional source information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Hits*  and  *Singles* contain information about the source, i.e. energy and  particle type (PDGEncoding). When an ion source is employed, instead of the information of the ion, the information associated with one of the particles emitted in the  decays can be of interest. An extra option has been included in the actor  that allows to specify the parentID of the particle that is  going to be considered as *source*. By default, this option is disabled. It can be enabled using the following command::

	/gate/actor/[Actor Name]/specifysourceParentID 0/1

When the option is enabled (it is set to 1), a text file must be included with a column of integers corresponding to the parentIDs  of the particles  that are going to be considered as primaries::

	/gate/actor/[Actor Name]/parentIDFileName  [text file name]

For example,  in the case of  a 22Na source, we are interested in the 1274 keV emitted gamma-ray and the annihilation photons that can be identified using a value for the parentID of 2 and 4 respectively (at least using livermore or em opt4 physics list).



Offline processing
------------------
Be aware that only .root extension output files can be processed offline.
The following executables:

* GateDigit_hits_digitizer
* GateDigit_singles_sorter
* GateDigit_coincidence_processor  

perform respectively an offline digitization, an offline sorter and an offline sequence coincidence reconstruction.
In order to use these executables during GATE compilation GATE_COMPILE_GATEDIGIT must be set to ON.     



