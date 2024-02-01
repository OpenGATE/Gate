.. _compton_camera_imaging_simulations-label:

Compton camera imaging simulations: CCMod (MOVED TO GENERAL DIGITIZER SINCE GATE 9.3)
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

For the DIGITIZER check the page: :ref:`_digitizer_and_readout_parameters-label`.



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



