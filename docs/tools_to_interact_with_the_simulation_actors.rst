.. _tools_to_interact_with_the_simulation_actors-label:

Tools to Interact with the Simulation : Actors
==============================================

.. contents:: Table of Contents
   :depth: 15
   :local:

Actors are tools which allow to interact with the simulation. They can collect information during the simulation, such as energy deposit, number of particles created in a given volume, etc. They can also modify the behavior of the simulation. Actors use hooks in the simulation : run (begin/end), event(begin/end), track(begin/end), step.

General Purpose
---------------

Add an actor
~~~~~~~~~~~~

Use the command::

   /gate/actor/addActor [Actor Type]  [Actor Name]

Attach to a volume
~~~~~~~~~~~~~~~~~~

Tells that the actor is attached to the volume [Volume Name]. For track and step levels, the actor is activated for step inside the volume and for tracks created in the volume. If no attach command is provided then the actor is activated in any volume. The children of the volume inherit the actor::

   /gate/actor/[Actor Name]/attachTo   [Volume Name]

Save output
~~~~~~~~~~~

This command allow to save the data of the actor to the file [File Name]. The particular behaviour (format, etc.) depends on the type of the actor::

   /gate/actor/[Actor Name]/save   [File Name]

It is possible to save the output every N events with the command::

   /gate/actor/[Actor Name]/saveEveryNEvents   [N]

It is possible to save the output every N seconds with the command:: 

  /gate/actor/[Actor Name]/saveEveryNSeconds [N]

3D matrix actor (Image actor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some actors, such as the :ref:`dose_measurement_doseactor-label`, can store some information into a 3D image (or matrix) according to the spatial position of the hit. User can specify the resolution of the 3D matrix (in this case, the size is equal to the size of the bounding box of the attached volume). Alternatively, user can specify the size to allow smaller matrices (never bigger).

* "attachTo" : the scoring value is stored in the 3D matrix only when a hit occur in the attached volume. If the size of the volume is greater than the 3D matrix, hit occurring out of the matrix are not recorded. Conversely, if the 3D matrix is larger than the attached volume, part which is outside the volume will never record hit (even if it occurs) because hit is performed out of the volume. 
* "type" : In Geant4, when a hit occurs, the energy is deposited along a step line. A step is defined by two positions the 'PreStep' and the 'PostStep'. The user can choose at which position the actor have to store the information (edep, dose ...) : it can be at PreStep ('pre'), at PostStep ('post'), at the middle between PreStep and PostStep ('middle') or distributed from PreStep to PostStep ('random'). According to the matrix size, such line can be located inside a single dosel or cross several dosels. Preferred type of hit is "random", meaning that a random position is computed along this step line and all the energy is deposited inside the dosel that contains this point. 
* the attached volume can be a voxelized image. The scoring matrix volume (dosels) are thus different from the geometric voxels describing the image::

   /gate/actor/[Actor Name]/attachTo       waterbox
   /gate/actor/[Actor Name]/setSize        5 5 5 cm
   /gate/actor/[Actor Name]/voxelsize      10 20 5 mm
   /gate/actor/[Actor Name]/setPosition    1 0 0 mm
   /gate/actor/[Actor Name]/stepHitType    random

* If you would like the dose actor to use exactly the same voxels as the input image, then the safest way to configure this is with *setResolution*. Otherwise, when setting *voxelsize*, rounding errors may cause the dosels to be slightly different, in particular in cases where the voxel size is not a nice round number (e.g. 1.03516 mm on a dimension with 512 voxels). Such undesired rounding effects have been observed Gate release 7.2 and may be fixed in a later release.

List of available Actors
------------------------

Simulation statistic
~~~~~~~~~~~~~~~~~~~~

This actor counts the number of steps, tracks, events, runs in the simulation. If the actor is attached to a volume, the actor counts the number of steps and tracks in the volume. The output is an ASCII file::

   /gate/actor/addActor SimulationStatisticActor     MyActor
   /gate/actor/MyActor/save                          MyOutput.txt

Electromagnetic (EM) properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This actor allows extracting EM properties for all materials defined in a simulation, as listed below:

* Density    (mass density in g/cm³)
* e-density  (electronic density in e-/mm³)
* RadLength  (radiation length in mm)
* I          (ionization potential in eV)
* EM-DEDX    (EM mass stopping power in MeV.cm²/g)
* Nucl-DEDX  (nuclear mass stopping power in MeV.cm²/g)
* Tot-DEDX   (total mass stopping power in MeV.cm²/g)

EM properties are calculated relative to a specific particle type and energy, as defined by the user. For instance, EM properties corresponding to a 30 MeV neutron can be calculated using the following command lines::

   /gate/actor/addActor EmCalculatorActor            MyActor
   /gate/actor/MyActor/setParticleName               proton
   /gate/actor/MyActor/setEnergy                     150 MeV
   /gate/actor/MyActor/save                          MyOutput.txt

.. _dose_measurement_doseactor-label:

Dose measurement (DoseActor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DoseActor builds 3D images of the energy deposited (edep), dose deposited and the number of hits in a given box volume (other types of volumes not supported; for a possible workaround see here `here <http://lists.opengatecollaboration.org/pipermail/gate-users/2022-May/013041.html>`_). It takes into account the weight of particles. It can store multiple information into a 3D grid, each information can be enabled by using::

   /gate/actor/[Actor Name]/enableEdep             true
   /gate/actor/[Actor Name]/enableUncertaintyEdep  true
   /gate/actor/[Actor Name]/enableSquaredEdep      true
   /gate/actor/[Actor Name]/enableDose             true
   /gate/actor/[Actor Name]/enableUncertaintyDose  true
   /gate/actor/[Actor Name]/enableDose             true
   /gate/actor/[Actor Name]/enableUncertaintyDose  true
   /gate/actor/[Actor Name]/enableSquaredDose      true
   /gate/actor/[Actor Name]/enableNumberOfHits     true

Informations can be disable by using "false" instead of "true" (by default all states are false)::

   /gate/actor/[Actor Name]/enableEdep             false

The unit of edep is MeV and the unit of dose is Gy. The dose/edep squared is used to calculate the uncertainty when the output from several files are added. The uncertainty is the relative statistical uncertainty. The "SquaredDose" flag allows to store the sum of squared dose (or energy). It is very useful when using GATE on several workstations with numerous jobs. To compute the final uncertainty, you only have to sum the dose map and the squared dose map to estimate the final uncertainty according to the uncertainty equations.

It is possible to normalize the maximum dose value to 1::

   /gate/actor/[Actor Name]/normaliseDoseToMax   true

For normalization purposes, further commands are also available::

   /gate/actor/[Actor Name]/normaliseDoseToWater   true

or::

   /gate/actor/[Actor Name]/normaliseDoseToIntegral   true


For the output, the suffixes Edep, Dose, NbOfHits, Edep-Uncertainty, Dose-Uncertainty, Edep-Squared or Dose-Squared are added to the output file name given by the user. You can use several files types: ASCII file (.txt), root file (.root), Analyze (.hdr/.img) and MetaImage (.mhd/.raw) (mhd is recommended!). The root file works only for 1D and 2D distributions::

   /gate/actor/addActor DoseActor             MyActor
   /gate/actor/MyActor/save                   MyOutputFile.mhd
   /gate/actor/MyActor/attachTo               MyVolume
   /gate/actor/MyActor/stepHitType            random
   /gate/actor/MyActor/setSize                5 5 5 m
   /gate/actor/MyActor/setResolution          1 1 3000 
   /gate/actor/MyActor/enableEdep             true
   /gate/actor/MyActor/enableUncertaintyEdep  true
   /gate/actor/MyActor/enableSquaredEdep      false
   /gate/actor/MyActor/enableDose             false
   /gate/actor/MyActor/normaliseDoseToMax     false

Water equivalent doses (or dose to water) can be also calculated, in order to estimate doses calculated using water equivalent path length approximations, such as in Treatment Planning Systems (TPS). The commands previously presented for the "dose" also work for the "dose to water" as shown below::

   /gate/actor/[Actor Name]/enableDoseToWater                   true
   /gate/actor/[Actor Name]/enableUncertaintyDoseToWater        true
   /gate/actor/[Actor Name]/normaliseDoseToWater                true

**New image format : MHD**

Gate now can read and write mhd/raw image file format. This format is similar to the previous hdr/img one but should solve a number of issues. To use it, just specify .mhd as extension instead of .hdr. The principal difference is that mhd stores the 'origin' of the image, which is the coordinate of the (0,0,0) pixel expressed in the *physical world* coordinate system (in general in millimetres). Typically, if you get a DICOM image and convert it into mhd (`vv <http://vv.creatis.insa-lyon.fr>`_ can conveniently do this), the mhd will keep the same pixel coordinate system as the DICOM. 

In GATE, if you specify the macro "TranslateTheImageAtThisIsoCenter" with the coordinate of the isocenter that is in a DICOM-RT-plan file, the image will be placed such that this isocenter is at position (0,0,0) of the mother volume (often the world). This is very useful to precisely position the image as indicated in a RT plan. Also, when using a DoseActor attached to a mhd file, the output dose distribution can be stored in mhd format. In this case, the origin of the dose distribution will be set such that it corresponds to the attached image (easy superimposition display).

Additional information can be found here: `here <http://lists.opengatecollaboration.org/pipermail/gate-users/2021-March/012331.html>`

Note however, that the mhd module is still experimental and not complete. It is thus possible that some mhd images cannot be read. Use and enjoy at your own risk, please contact us if you find bugs and be warmly acknowledged if you correct bugs.

Dose by regions
^^^^^^^^^^^^^^^

The dose actor can also calculate dose and energy deposited in regions defined by a set of voxels and outputs the result in a text file. These regions are read from a .mhd image file containing labels (integers) which must be of the same size as the dose actor. Each label in the image defines a region where all energies will be summed and the dose calculated during the simulation. A region must contain voxels of the same material for the dose calculation to be correct. This output allows to get the statistical uncertainties for a set of voxels.

To activate this output::

   /gate/actor/[Actor Name]/inputDoseByRegions     data/regionImage.mhd
   /gate/actor/[Actor Name]/outputDoseByRegions    output/DoseByRegions.txt

It is possible to define additional regions composed of original regions (of the same material) by specifying a new region label followed by a colon and the list of original region labels::

   /gate/actor/[Actor Name]/addRegion              1000: 89, 90, 91
   /gate/actor/[Actor Name]/addRegion              1001: 92, 93, 94

The output ascii file contains one line per region with the following information::

   #id 	vol(mm3) 	edep(MeV) 	std_edep 	sq_edep 	dose(Gy) 	std_dose 	sq_dose 	n_hits 	n_event_hits
   0	158092650.2908	13.08421506078	0.053474625991	0.489560086787	1.10061390e-11	0.053474625991	3.46402200e-25	40288	814

An example can be found in the GateContrib GitHub repository under `dosimetry/doseByRegions <https://github.com/OpenGATE/GateContrib/tree/master/dosimetry/doseByRegions>`_.

Dose calculation algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When storing a dose (D=edep/mass) with the DoseActor, mass is computed by using the material density at the step location and using the volume of the dosel. If the size of the image voxel is smaller than the size of the dosel of the DoseActor it can lead to undesired results. Two algorithms are available for the DoseActor.

Volume weighting algorithm
++++++++++++++++++++++++++

This algorithm is used by default. The absorbed dose of each material is ponderated by the fraction of materials volume::

   /gate/actor/[Actor Name]/setDoseAlgorithm VolumeWeighting

Mass weighting algorithm
++++++++++++++++++++++++

This algorithm calculates the dose of each dosel by taking the deposited energy and dividing it by its mass:: 

  /gate/actor/[Actor Name]/setDoseAlgorithm MassWeighting

**Mass image :**

Mass images (.txt, .root, .mhd) can be imported and exported to be used by the mass weighting algorithm.

* Export::

   /gate/actor/[Actor Name]/exportMassImage path/to/MassImage

* Import::

   /gate/actor/[Actor Name]/importMassImage path/to/MassImage

* The unit of mass images is kg.
* When the mass weighting algorithm is used on a unvoxelized volume, depending on the dosel's resolution of the DoseActor the computation can take a very long time. 
* **Important note :** If no mass image is imported when using the mass weighting algorithm Gate will calculate the mass during the simulation (this can take a lot of time).

The command 'exportMassImage' can be used to generate the mass image of the DoseActor's attached volume one time for all and import it with the 'importMassFile' command.
 
**Limitations :**

* **With voxelized phantom :**

  - The MassWeighting algorithm works with phantoms imported with *ImageRegularParametrisedVolume* and *ImageNestedParametrisedVolume*.
  - For now it's not possible to choose an actor resolution smaller than the phantom's resolution.
  - It is mandatory to attach the actor directly to the phantom.

* **With unvoxelized geometry :** The dosel's resolution must be reasonably low otherwise the time of calculation can be excessively long! (and can need a lot of memory!)

Tet-Mesh Dose Actor
~~~~~~~~~~~~~~~~~~~

The **TetMeshDoseActor** can only be attached to 'TetMeshBox' volumes. It scores dose for each tetrahedron of the tetrahedral mesh contained in the TetMeshBox. Example usage::

   /gate/actor/addActor              TetMeshDoseActor doseSensor
   /gate/actor/doseSensor/attachTo   meshPhantom
   /gate/actor/doseSensor/save       output/phantom_dose.csv

The output of the TetMeshDoseActor is a csv-file tabulating the results, e.g.::

    # Tetrahedron-ID, Dose [Gy], Relative Uncertainty, Sum of Squared Dose [Gy^2], Volume [cm^3], Density [g / cm^3], Region Marker
    0, 1.33e-08, 1.30e-01, 3.03e-18, 1.94e-02, 9.49e-01, 1
    1, 1.96e-09, 9.99e-01, 3.86e-18, 1.13e-04, 9.49e-01, 1
    ...

Each row corresponds to one tetrahedron. The region marker column identifies to which macroscopic structure a tetrahedron belongs to -- it is equal to the region attribute defined for this tetrahedron in the '.ele' file the TetMeshBox is constructed from.

.. _biodose_measurement_biodoseactor-label:

Biological dose measurement (BioDoseActor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The BioDoseActor builds 3D images of::

   the energy deposited (edep)
   physical and biological doses deposited
   alphamix and sqrtbetamix deposited
   the number of hits

in a given box volume.
The biological dose is computed as::

   :math:`\frac{-\alpha_{ref} + \sqrt{\alpha_{ref}^2 + 4\,\beta_{ref}\,({\alpha_{mix}}\,Z + ({\sqrt{\beta_{mix}}}\,Z)^2 )}}{2\,\beta_{ref}}`

It takes into account the weight of particles.
It can store multiple information into a 3D grid, each information can be enabled by using::

   /gate/actor/[Actor Name]/enableEdep                true
   /gate/actor/[Actor Name]/enableDose                true
   /gate/actor/[Actor Name]/enableAlphaMix            true
   /gate/actor/[Actor Name]/enableSqrtBetaMix         true
   /gate/actor/[Actor Name]/enableRBE                 true
   /gate/actor/[Actor Name]/enableUncertainty         true
   /gate/actor/[Actor Name]/enableUncertaintyDetails  true
   /gate/actor/[Actor Name]/enableHitEventCount       true

Informations can be disabled by using "false" (default) instead of "true".
The unit of edep is MeV, the unit of dose is Gy, the unit of biological dose is Gy RBE.
The uncertainty outputs are relative statistical uncertainties.

For the output, the suffixes::

   _edep
   _dose
   _alphamix
   _sqrtbetamix
   _biodose
   _rbe
   _hitevent_count
   _dose_uncertainty
   _biodose_uncertainty

are added to the output file name given by the user.

The user must provide a cell line and a biophysical model::

   /gate/actor/[Actor Name]/setCellLine         HSG
   /gate/actor/[Actor Name]/setBioPhysicalModel NanOx

which will be combined to make a filename: data/{CellLine}_{BioPhysicalModel}.db, for example data/HSG_NanOx.db.
Only HSG (Human Salivary Gland) cell line is provided.
Two biophysical models are available: NanOx (carbon, proton) and MMKM (carbon).
User can provide its own data files (cell line associated to a biophysical model database).

The user must also provide the alpha and beta reference values corresponding to the cell line::

   /gate/actor/[Actor Name]/setAlphaRef         0.313
   /gate/actor/[Actor Name]/setBetaRef          0.0615

If the user wants to reach a large dose value output to 
The user can use a dose scale factor in order to reach a large dose value without needing to run higher number of primary particles.
The dose scale factor is set with this command (default 1)::

   /gate/actor/[Actor Name]/setDoseScaleFactor  1e3

It will multiply the dose value (and affect the biological dose value).
Note that it will not affect the relative uncertainty, so it can be used once some uncertainty threshold has been reached.

Usage example::

   /gate/actor/addActor                    BioDoseActor MyBio
   /gate/actor/MyBio/attachTo              MyVolume
   /gate/actor/MyBio/setVoxelSize          200 200 1 mm
   /gate/actor/MyBio/setPosition           0 0 0
   /gate/actor/MyBio/setCellLine           HSG
   /gate/actor/MyBio/setBioPhysicalModel   NanOx
   /gate/actor/MyBio/setAlphaRef           0.313
   /gate/actor/MyBio/setBetaRef            0.0615
   /gate/actor/MyBio/enableDose            true
   /gate/actor/MyBio/save                  output/{particleName}.mhd

.. _kill_track-label:

Kill track
~~~~~~~~~~

This actor kills tracks entering the volume. The output is the number of tracks killed. It is stored an ASCII file::

   /gate/actor/addActor KillActor       MyActor
   /gate/actor/MyActor/save             MyOutputFile.txt
   /gate/actor/MyActor/attachTo         MyVolume

Stop on script
~~~~~~~~~~~~~~

This actor gets the output of a script and stop the simulation if this output is true::

   /gate/actor/addActor  StopOnScriptActor     MyActor
   /gate/actor/MyActor/save                    MyScript

It is possible to save all the other actors before stopping the simulation with the command::

   /gate/actor/MyActor/saveAllActors           true

Track length
~~~~~~~~~~~~

This actor stores the length of each tracks in a root file. It takes into account the weight of particles. They are three commands to define the boundaries and the binning of the histogram::

   /gate/actor/addActor  TrackLengthActor      MyActor
   /gate/actor/MyActor/save                    MyOutputFile.root
   /gate/actor/MyActor/setLmin                 0 mm
   /gate/actor/MyActor/setLmax                 1 cm
   /gate/actor/MyActor/setNumberOfBins         200

Energy spectrum
~~~~~~~~~~~~~~~

This actor builds one file containing N histograms. By default 3 histograms are enabled: The fluence and energy deposition spectra differential in energy and the energy deposition spectrum as a function of LET. Ideally one specifies the lower (Emin) and upper (Emax) boundary of the histogram and the resolution/number of bins::

   /gate/actor/addActor  EnergySpectrumActor                MyActor
   /gate/actor/MyActor/save                                 MyOutputFile.root
   /gate/actor/MyActor/energySpectrum/setEmin               0 eV
   /gate/actor/MyActor/energySpectrum/setEmax               200 MeV
   /gate/actor/MyActor/energySpectrum/setNumberOfBins       2000

   /gate/actor/MyActor/enableLETSpectrum				            true
   /gate/actor/MyActor/LETSpectrum/setLETmin			          0 keV/um
   /gate/actor/MyActor/LETSpectrum/setLETmax			          100 keV/um
   /gate/actor/MyActor/LETSpectrum/setNumberOfBins			    1000
   
   /gate/actor/MyActor/energyLossHisto/setEdepMin               0.0001 keV 
   /gate/actor/MyActor/energyLossHisto/setEdepMax               200 keV
   /gate/actor/MyActor/energyLossHisto/setNumberOfEdepBins       1000    

By default an equidistant bin width is applied. However, a logarithmic bin width may be enabled::

   /gate/actor/MyActor/setLogBinWidth                   true

In that case the lower boundary of the histogram should not be 0. If 0 is specified as lower boundary, it is replaced with a :math:`\epsilon` > 0 internally. 

To normalize the 1D histograms to the number of simulated primary events enable::

   /gate/actor/MyActor/normalizeToNbPrimaryEvents                   true

To score the energy relative to unit particle mass [MeV/u] instead of total energy [MeV] enable::

   /gate/actor/MyActor/setEnergyPerUnitMass                   true

The number of particles entering a volume differential in energy: (this is not fluence)::

   /gate/actor/MyActor/enableNbPartSpectrum			true

The fluence differential in energy corrected by 1/cos(:math:`\phi`) with :math:`\phi` being the angle of the particle entering a volume. This works only for planes perpendicular to the z direction. No correction for cos(:math:`\phi`) = 0 is applied. Only particles entering the volume are scored::

 /gate/actor/MyActor/enableFluenceCosSpectrum			true

The fluence differential in energy summing up the track length of the particle. The outcome of this vector needs to be divided by the volume of the geometry the actor was attached to::

 /gate/actor/MyActor/enableFluenceTrackSpectrum			true

The energy deposition differential in energy is scored using GetTotalEnergyDeposit()::

 /gate/actor/MyActor/enableEdepSpectrum			true

the energy deposition per event ('edepHisto'), the energy deposition per track ('edepTrackHisto') and the energy loss per track ('eLossHisto') and the energy deposition per step ('edepStepHisto'). These histograms are stored in a root file. They take into account the weight of particles::

   /gate/actor/MyActor/enableEdepHisto		true
   /gate/actor/MyActor/enableEdepTimeHisto		true
   /gate/actor/MyActor/enableEdepTrackHisto		true
   /gate/actor/MyActor/enableEdepStepHisto		true
   /gate/actor/MyActor/enableElossHisto		true
   /gate/actor/MyActor/energyLossHisto/setEmin              0 eV
   /gate/actor/MyActor/energyLossHisto/setEmax              15 MeV
   /gate/actor/MyActor/energyLossHisto/setNumberOfBins      120

To score the energy deposition differential in :math:`Q = charge^2 / E_{kin}`::

   /gate/actor/MyActor/enableQSpectrum					true
   /gate/actor/MyActor/QSpectrum/setQmin				0 keV/um
   /gate/actor/MyActor/QSpectrum/setQmax				100 keV/um
   /gate/actor/MyActor/QSpectrum/setNumberOfBins			1000

By default histograms are saved as .root files. The histograms will be (in addition) converted to ASCII format files by enabling::

   /gate/actor/MyActor/saveAsText				true

Production and stopping particle position
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This actor stores in a 3D image the position where particles are produced and where particles are stopped. For the output, the suffixes 'Prod' and 'Stop' are added to the output file name given by the user. You can use several files types: ASCII file (.txt), root file (.root), (.mhd/.raw or .hdr/.img). The root file works only for 1D and 2D distribution::

   /gate/actor/addActor ProductionAndStoppingActor      MyActor
   /gate/actor/MyActor/save                             MyOutputFile.mhd
   /gate/actor/MyActor/attachTo                         MyVolume
   /gate/actor/MyActor/setResolution                    10 10 100
   /gate/actor/MyActor/stepHitType                      post

**< ! >  In Geant4, secondary production occurs at the end of the step, the recommended state for 'stepHitType' is 'post'**

The "prod" output contains the 3D distribution of the location where particles are created (their first step), and the "stop" contains the 3D distribution of the location where particles stop (end of track). Each voxel of both images thus contains the number of particles that was produced (resp. stopped) in this voxel. Source code is: https://github.com/OpenGATE/Gate/blob/develop/source/digits_hits/src/GateProductionAndStoppingActor.cc

Secondary production
~~~~~~~~~~~~~~~~~~~~

This actor creates a root file and stores the number of secondaries in function of the particle type. Ionisation electrons are dissociated from electrons produced by other processes. Decay positrons are dissociated from positrons produced by other processes. Gammas are classified in four categories: gammas produced by EM processes, gammas produced by hadronic processes, gammas produced by decay processes and other gammas::

   /gate/actor/addActor  SecondaryProductionActor     MyActor
   /gate/actor/MyActor/save                           MyOutputFile.root
   /gate/actor/MyActor/attachTo                       MyVolume

Delta kinetic energy
~~~~~~~~~~~~~~~~~~~~

This actor sums the relative and absolute :math:`\Delta` (kinetic energy) and stores the results in two files (with suffixes "-RelStopPower" and "-StopPower"). It also stores the map of the hits to allow users to calculate the mean values::

   /gate/actor/addActor   StoppingPowerActor       MyActor
   /gate/actor/MyActor/save                        MyOutputFile.hdr
   /gate/actor/MyActor/attachTo                    MyVolume
   /gate/actor/MyActor/setResolution               10 10 100
   /gate/actor/MyActor/stepHitType                 random

Number of particles entering volume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This actor builds a map of the number of particules produced outside of the actor volume and interacting in the volume. The particle is recorded once in each voxel where it interacting::

   /gate/actor/addActor    ParticleInVolumeActor       MyActor
   /gate/actor/MyActor/save                            MyOutputFile.hdr
   /gate/actor/MyActor/attachTo                        MyVolume
   /gate/actor/MyActor/setResolution                   10 10 100
   /gate/actor/MyActor/stepHitType                     post

Q-value
~~~~~~~

This actor calculates the Q-values of interactions::

   /gate/actor/addActor     QvalueActor         MyActor
   /gate/actor/MyActor/save                     MyOutputFile.hdr
   /gate/actor/MyActor/attachTo                 MyVolume
   /gate/actor/MyActor/setResolution            10 10 100
   /gate/actor/MyActor/stepHitType              random


CrossSectionProductionActor
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CrossSectionProductionActor derives the production of C-11 or O-15 from the equation proposed by (Parodi et al, 2007). The cross section data are provided directly in the class code. By default, only the production of the C-11 is activated.
 
**WARNING**: The size of the image has to be given in mm

The current limit in cross section data is 199 MeV. Other data can be added in the class::

   /gate/actor/addActor                CrossSectionProductionActor beta
   /gate/actor/beta/attachTo           volume
   /gate/actor/beta/save               output_dump/test_small.hdr
   /gate/actor/beta/addO15             true
   /gate/actor/beta/addC11             true
   /gate/actor/beta/setVoxelSize       1 1 1 mm
   /gate/actor/beta/saveEveryNEvents   100000


WashOutActor
~~~~~~~~~~~~

The bilogical washout follows the Mizuno model (H. Mizuno et al. Phys. Med. Biol. 48, 2003). The activity distributions of the washout actor associated volume are continuously modified as a function of the acquisition time in terms of the following equation :

:math:`Cwashout(t)=Mf.exp(-t/Tf.ln2)+Mm.exp(-t/Tm.ln2)+Ms.exp(-t/Ts.ln2)`

Where 3 components are defined (fast, medium and slow) with two parameters for each : the half life T and the fraction M (Mf + Mm + Ms = 1). 

Users should provide a table as an ASCII file with the washout parameters values for any radioactive source in the associated volume. In order to take into account the physiological properties of each tissue, it is important to highlight that one independent radioactive source should be defined per each material involved in the simulation::

   /gate/actor/addActor                               WashOutActor [ACTOR NAME]
   /gate/actor/[ACTOR NAME]/attachTo    	           [VOLUME NAME]
   /gate/actor/[ACTOR NAME]/readTable		   [TABLE FILE NAME]

Example of [TABLE FILE NAME]: How to specify different parameters which are associated to the washout model - This ASCII file will be used by the washout Actor::

   2 
   [SOURCE 1 NAME]   [MATERIAL 1 NAME]     [Mf VALUE]  [Tf VALUE IN SEC]   [Mm VALUE]  [Tm VALUE IN SEC]   [Ms VALUE]  [Ts VALUE IN SEC] 
   [SOURCE 2 NAME]   [MATERIAL 2 NAME]     [Mf VALUE]  [Tf VALUE IN SEC]   [Mm VALUE]  [Tm VALUE IN SEC]   [Ms VALUE]  [Ts VALUE IN SEC] 
   ...
   ...


Fluence Actor (particle counting)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This actor counts the number of time a (new) particle is passing through a volume; output as an image::

   /gate/actor/addActor FluenceActor      Detector
   /gate/actor/Detector/save              output/detector.mhd
   /gate/actor/Detector/attachTo          DetectorPlane
   /gate/actor/Detector/stepHitType       pre
   /gate/actor/Detector/setSize           10 410 410 mm
   /gate/actor/Detector/setResolution     1 256 256
   /gate/actor/Detector/enableScatter     true

.. _tle_and_setle_track_length_estimator-label:

TLE and seTLE (Track Length Estimator)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TLE is the Track Length Estimator method initially proposed by [Williamson1997] allowing very fast dose computation for low energy photon beam (about below 1 MeV). About 1000x faster than analog Monte-Carlo. The second method, seTLE for split-exponential TLE, was proposed in [Smekens2014] and is about 15x faster than TLE. 

* Williamson J F 1987 Monte Carlo evaluation of kerma at a point for photon transport problems Med. Phys. 14 567–76
* F. Smekens, J. M. Létang, C. Noblet, S. Chiavassa, G. Delpon, N. Freud, S. Rit, and D. Sarrut, "Split exponential track length estimator for Monte-Carlo simulations of small-animal radiation therapy", Physics in medicine and biology, vol. 59, issue 24, pp. 7703-7715, 2014 `pdf <http://iopscience.iop.org/0031-9155/59/24/7703/pdf/0031-9155_59_24_7703.pdf>`_
* F. Baldacci, A. Mittone, A. Bravin, P. Coan, F. Delaire, C. Ferrero, S. Gasilov, J. M. Létang, D. Sarrut, F. Smekens, et al., "A track length estimator method for dose calculations in low-energy x-ray irradiations: implementation, properties and performance", Zeitschrift Fur Medizinische Physik, 2014.
* A. Mittone, F. Baldacci, A. Bravin, E. Brun, F. Delaire, C. Ferrero, S. Gasilov, N. Freud, J. M. Létang, D. Sarrut, et al., "An efficient numerical tool for dose deposition prediction applied to synchrotron medical imaging and radiation therapy.", Journal of synchrotron radiation, vol. 20, issue Pt 5, pp. 785-92, 2013

Usage is very simple just replace the DoseActor by TLEDoseActor. See examples/example_Radiotherapy/example10 in the Gate source code::

   /gate/actor/addActor                  TLEDoseActor  tle
   /gate/actor/tle/attachTo    	      phantom
   /gate/actor/tle/stepHitType           random
   /gate/actor/tle/setVoxelSize          2 2 2 mm
   /gate/actor/tle/enableDose            true
   /gate/actor/tle/save                  output/dose-tle.mhd

or::

   /gate/actor/addActor                             SETLEDoseActor setle
   /gate/actor/setle/attachTo                       phantom
   /gate/actor/setle/setVoxelSize                   2 2 2 mm
   /gate/actor/setle/enableHybridino                true
   /gate/actor/setle/setPrimaryMultiplicity         200
   /gate/actor/setle/setSecondaryMultiplicity       400
   /gate/actor/setle/enableDose                     true
   /gate/actor/setle/save                           output/dose-setle.mhd

A detailed documentation is available here: http://midas3.kitware.com/midas/download/item/316877/seTLE.pdf


Fixed Forced Detection CT
~~~~~~~~~~~~~~~~~~~~~~~~~

This actor is a *Variance Reduction Technique* for the simulation of CT.

The fixed forced detection technique (Colijn & Beekman 2004, Freud et al. 2005, Poludniowski et al. 2009) relies on the deterministic computation of the probability of the scattered photons to be aimed at each pixel of the detector. The image of scattered photons is obtained from the sum of these probabilities.

The probability of each scattering point to contribute to the center of the j−th pixel is the product of two terms:

* the probability of the photon to be scattered in the direction of the pixel
* the probability of the scattered photon to reach the detector and to be detected

**Fixed Forced Detection summary**

1) Deterministic simulation of the primary (DRR)
2) Low statistics Monte Carlo simulation ⇒ Compute scattering points
3) Fixed forced detection (deterministic)

Inputs::

   /gate/actor/addActor    FixedForcedDetectionActor        MyActor
   /gate/actor/MyActor/attachTo                             world
   /gate/actor/MyActor/setDetector                          DetectorPlane
   /gate/actor/MyActor/setDetectorResolution                128 128
   /gate/actor/MyActor/responseDetectorFilename             responseDetector.txt

The detector response δ(E) is modeled with a continuous energy-function that describes the average measured signal for a given incident energy E. The output signal in each image depends on the detector response (parameter responseDetectorFilename). For examples, if δ(E)=1, then the output signal is the number of photons, and if δ(E)=E (as responseDetector.txt in the github example), then the output signal is the total energy of photons.

One can separate compton, rayleigh and fluorescence photons, secondary (compton+rayleigh+fluorescence), primary or total (secondary+primary). flatfield is available to compute the measured primary signal if there is no object, which is useful for CT to apply the Beer Lambert law. The attenuation is ln(flatfield/primary) to get the line integral, i.e., the input of most CT reconstruction algorithms. To include the secondary signal (compton+rayleigh+fluorescence) in the attenuation, one can use the images saved by the actor to recompute the attenuation (for example using ITK in Python). The formula for the attenuation would be ln(flatfield / (primary+secondary)).

* **attachTo** ⇒ Attaches the sensor to the given volume
* **saveEveryNEvents** ⇒ Save sensor every n Events.
* **saveEveryNSeconds** ⇒ Save sensor every n seconds.
* **addFilter** ⇒ Add a new filter
* **setDetector** ⇒ Set the name of the volume used for detector (must be a Box).
* **setDetectorResolution** ⇒ Set the resolution of the detector (2D).
* **geometryFilename** ⇒ Set the file name for the output RTK geometry filename corresponding to primary projections.
* **primaryFilename** ⇒ Set the file name for the primary x-rays (printf format with runId as a single parameter).
* **materialMuFilename** ⇒ Set the file name for the attenuation lookup table. Two paramaters: material index and energy.
* **attenuationFilename** ⇒ Set the file name for the attenuation image (printf format with runId as a single parameter).
* **responseDetectorFilename** ⇒ Input response detector curve.
* **flatFieldFilename** ⇒ Set the file name for the flat field image (printf format with runId as a single parameter).
* **comptonFilename** ⇒ Set the file name for the Compton image (printf format with runId as a single parameter).
* **rayleighFilename** ⇒ Set the file name for the Rayleigh image (printf format with runId as a single parameter).
* **fluorescenceFilename** ⇒ Set the file name for the fluorescence image (printf format with runId as a single parameter).
* **secondaryFilename** ⇒ Set the file name for the scatter image (printf format with runId as a single parameter).
* **enableSquaredSecondary** ⇒ Enable squared secondary computation
* **enableUncertaintySecondary** ⇒ Enable uncertainty secondary computation
* **totalFilename** ⇒ Set the file name for the total (primary + scatter) image (printf format with runId as a single parameter).
* **phaseSpaceFilename** ⇒ Set the file name for storing all interactions in a phase space file in root format.
* **setInputRTKGeometryFilename** ⇒ Set filename for using an RTK geometry file as input geometry.
* **noisePrimaryNumber** ⇒ Set a number of primary for noise estimate in a phase space file in root format.
* **energyResolvedBinSize**  ⇒ Set energy bin size for having an energy resolved output. Default is 0, i.e., off.

An example is available at example_CT/fixedForcedDetectionCT.

The GateHybridForcedDetectionActor works for:

* One voxelized (CT) volume, the rest must be of the same material as the world → No volume between voxelized volume and detector.
* Point sources (plane distribution focused).
* A given detector description.
* With some additional geometric limitations.

The FFD implementation in Gate is based on the Reconstruction Toolkit. The deterministic part, the ray casting, is multi-threaded. One can control the number of threads by setting the environment variable ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS. If it is not set, the default is to have a many threads as cores in the machine.

Fixed Forced Detection CT with Fresnel phase contrast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provided that you also compile Gate with GATE_USE_XRAYLIB ON (in addition to RTK), i.e., that you activate the dependency to the `xraylib <https://github.com/tschoonj/xraylib>`_, you can also account for the change of phase in the x-ray wave in the computation of primary images with the following options:

* **materialDeltaFilename** ⇒ Set the output file name for the refractive index decrement lookup table. Two paramaters: material index and energy.
* **fresnelFilename** ⇒ Set the output file name for the Fresnel diffraction image (printf format with runId as a single parameter).

The output in fresnelFilename is computed following equation (2) of `Weber et al, Journal of Microscopy, 2018 <http://doi.org/10.1111/jmi.12606>`_.

An example is available at `GateContrib: Fresnel_FFD <https://github.com/OpenGATE/GateContrib/tree/master/imaging/CT/Fresnel_FFD>`_.

Fixed Forced Detection SPECT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This actor is a *Variance Reduction Technique* for the simulation of SPECT.

The fixed forced detection technique (Colijn & Beekman 2004, Freud et al. 2005, Poludniowski et al. 2009) relies on the deterministic computation of the probability of the scattered photons to be aimed at each pixel of the detector. The image of scattered photons is obtained from the sum of these probabilities.

The probability of each scattering point to contribute to the center of the j−th pixel is the product of two terms:

* the probability of the photon to be scattered in the direction of the pixel
* the probability of the scattered photon to reach the detector and to be detected

Inputs::

   /gate/actor/addActor    FixedForcedDetectionActor        MyActor
   /gate/actor/MyActor/attachTo                             world
   /gate/actor/MyActor/setDetector                          DetectorPlane
   /gate/actor/MyActor/setDetectorResolution                128 128
   /gate/actor/MyActor/setSourceType                        isotropic
   /gate/actor/MyActor/generatePhotons 	                    true

or::

   /gate/actor/MyActor/connectARF 	                         true

* **attachTo** ⇒ Attaches the sensor to the given volume
* **setDetector** ⇒ Set the name of the volume used for detector (must be a Box).
* **setDetectorResolution** ⇒ Set the resolution of the detector (2D).
* **generatePhotons** ⇒ Generates weighted photons outside of the volume directed at each pixel of the detector.
* **connectARF** ⇒ Connects the output of the FFD to ARF tables (see :ref:`angular_response_functions_to_speed-up_planar_or_spect_simulations-label`).

An example is available at `GateContrib: SPECT_FFD <https://github.com/OpenGATE/GateContrib/tree/master/imaging/SPECT_FFD>`_

The GateHybridForcedDetectionActor works for:

* One voxelized (CT) volume, the rest must be of the same material as the world → No volume between voxelized volume and detector.
* With some additional geometric limitations.

PromptGammaTLEActor
~~~~~~~~~~~~~~~~~~~

This actor is used to investigate prompt gamma production in proton therapy simulations. It provides a speedup factor of around 1000 compared to analog MC. vpgTLE is broken up into three parts. Stage 0 is required to be run once, and each vpgTLE simulation is then broken up into Stage 1 and Stage 2. For each stage, you can find and example in the *examples/vpgTLE* directory.

To understand the background, physics and mathematics of this example, refer to *Accelerated Prompt Gamma estimation for clinical Proton Therapy simulations* by B.F.B. Huisman.


LET Actor
~~~~~~~~~

This actor calculates the dose or track averaged linear energy transfer in units of keV/um::

   /gate/actor/addActor    LETActor       MyActor
   /gate/actor/MyActor/save               output/myLETactor.mhd
   /gate/actor/MyActor/attachTo           phantom
   /gate/actor/MyActor/setResolution      1 1 100
   /gate/actor/MyActor/setType            DoseAveraged

Options: DoseAveraged (default) or TrackAveraged. The implementation is equivalent to "Method C" in 'Cortes-Giraldo and Carabe, 2014, A critical study on different Monte Carlo scoring method of dose-average-linear energy transfer maps.' The stopping power is retrieved from the Geant4 EMCalculator method "ComputeElectronicDEDX". 
Method "A" could be enabled by setting the type to "DoseAveragedEdep", but this method is not recommended as it is not benchmarked and suffers from interplay effects with the geometric boundaries (voxel boundaries), step limiter and production cut values. 

Merging several independent LET simulations (e.g. from runs on several CPUs on a cluster) requires special care. Track/Fluence and Dose- averaged LET are calculated from the quotient of a sum of weighted LET values and normalized by the sum of step lenghts or energy deposition. Merging several simulations requires that the numerator and denominator are summed up individually and those sums are divided. Therefore, for splitting the simulation into sevaral sub-simulations (e.g. parallel computation) enable::

   /gate/actor/MyActor/doParallelCalculation true

The default value is false. Enabling this option will produce 2 output images for each LET actor and run, a file labeled as '-numerator' and one labeled as '-denominator'. Building the quotient of these two images results in the averaged LET image. Note that the numerator and denominator images have to be summed up before the division. The denominator file equals the dose and fluence if DoseAveraged and TrackAveraged is enabled, respectively, after normalizing by the mass or volume.

By default, the stopping power of the material at the PreStepPoint is used. Often a conversion to the LET (in particular water) is of interest. To convert the stopping power to another material than present in the volume use::

   /gate/actor/MyActor/setOtherMaterial G4_WATER
   
It may be of interest to separate the LET into several regions. Using following commands
   /gate/actor/MyActor/setLETthresholdMin 10 keV/um
   /gate/actor/MyActor/setLETthresholdMax 100 keV/um
will only score particles having a LET between 10 and 100 keV/um. In this way the average LETd,t in that region can be extracted. Note, when enabling the doParallelCalculation option also the dose and fluence of particles of particles with a certain LET can be extracted.

ID and particle filters can be used::

   /gate/actor/MyActor/addFilter                    particleFilter
   /gate/actor/MyActor/particleFilter/addParticle   proton
   
Other options in the LET Actor are to calculate the fluence averaged kinetic Energy enabled by changing the setType to "AverageKinEnergy". Radiochromic EBT3 films suffer from a LET dependent response in proton beam radiotherapy. One correction method has been proposed in Resch et al. 2019, the $g_{Q,Q0}$ factors can be calculated setting the type to "gqq0EBT3linear" or "gqq0EBT3fourth", which enables the first or fourth order polynomial correction function, respectively. 


Tissue Equivalent Proportional Counter Actor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The   tissue-equivalent   proportional   counter   (TEPC)   is   a   detector   dedicated   to   the measurement  of  the  lineal  energy  transfer  (LET)  distribution in  a volume  at  the  micrometric  scale.These physical data, depending on the beam quality and the location of the detector in the beam, is mainly  used  to  calculate  the biological  dose  for  high  LET  radiation  and  to characterize  the  beam quality for radioprotection issues.

A TEPC is very similar to a classical gas ionization chamber. The major difference relies in the sensible volume,  which  is  spherical  and  filled  with  low  pressure tissue
-equivalent  gas  instead  of  air. These characteristics  allow the  TEPC  to  mimic the  shape  and  composition  of  the  tiny  structures  in  a  cell nucleus (about 1 μm of diameter).

**Quick use**

The  principle of  the  TEPCactor  is  the  same  as  the  EnergySpectrumActor,  except that  the frequency  of  lineal  energy  is  stored  instead  of  the  deposited  energy.  In  order  to  obtain  the  lineal energy,  the  deposited  energy  is  divided  by  the  mean  chord  of  the  TEPC  volume (:math:`\overline{L}=\frac{2}{3}\pi\varnothing_{TEPC}`). This imposes creating a sphere as geometry for the TEPC.

**Generic commands** – The following commands allow to create, attach and save the result in a ROOT file (and a .txt file, if necessary)::

   /gate/actor/addActor TEPCActor  myTEPC
   /gate/actor/myTEPC/attachTo     myDetector
   /gate/actor/myTEPC/saveAsText   true
   /gate/actor/myTEPC/save         output/myLETspectrum.root

**Pressure command** – The pressure  of the tissue-equivalent gas (propane-based material) is used to tune the size of the water equivalent sphere represented by the TEPC detector.
In the literature, the density  of  such  materials  is  generally  defined  for  standard  pressure  and  temperature  conditions. Although the   user   can directly   create a low   pressure   and   density   gas material in   the “data/myGateMaterial.db” file, the following  command  allows  to modify  in-line  the  pressure  in  the TEPC material
if this one is defined for standard pressure and temperature conditions::

   /gate/actor/myTEPC/setPressure   0.044 bar

**Output commands** – This  list  of  commands makes  it  possible to  change  the  scale  of  the LET distribution  in  order  to  correctly  fit  with  the expected  results.  As  the  lineal  energy  distribution generally extends on several orders of magnitude, the default option is the logarithmic scale::

   /gate/actor/myTEPC/setLogscale     true
   /gate/actor/myTEPC/setNumberOfBins 150
   /gate/actor/myTEPC/setEmin         0.01 keV
   /gate/actor/myTEPC/setNOrders      6

This could be replaced by a linear scale::

   /gate/actor/myTEPC/setLogscale      false
   /gate/actor/myTEPC/setNumberOfBins  150
   /gate/actor/myTEPC/setEmin          0 keV
   /gate/actor/myTEPC/setEmax          100 keV

The last command allows to normalize the distribution by the number of incident particles::

   /gate/actor/myTEPC/setNormByEvent   true

**Example**

An  example  of  a TEPC actor  use  is  provided  in  the  example repository under `dosimetry/TEPCActor <https://github.com/OpenGATE/GateContrib/tree/master/dosimetry/TEPCActor>`_ folder.  In  this  example,  a  TEPC detector  is placed at different  positions  in  a  water  tank  and  irradiated  with a 155  MeV mono-energetic proton beam. This setup was used to validate the results against  the  TEPC measurements published  by  Kase  et  al.  2013. In  this  comparison,  our key  point  was  the  optimization  of  the particle cuts and step limiters. Indeed, the lineal energy distribution at the micrometric scale is highly sensible to these  two parameters. The particle cuts must be  low  enough to simulate  any significant contribution in the lineal energy distribution and the step limiters must bec correctly tuned in order to avoid  boundary  effects  on  geometry  elements,  while  keeping  the  global  simulation  time  as  low  as possible. More information regarding the geometry and the physical parameters that were tested to obtain the final macro files are available in the example repository (`TEPCactor.pdf <https://github.com/OpenGATE/GateContrib/blob/master/dosimetry/TEPCActor/TEPCactor.pdf>`_).

Phase Space Actor
~~~~~~~~~~~~~~~~~

Example::

   /gate/actor/addActor PhaseSpaceActor         MyActor
   /gate/actor/MyActor/save                     MyOutputFile.IAEAphsp
   /gate/actor/MyActor/attachTo                 MyVolume
   /gate/actor/MyActor/enableProductionProcess  false
   /gate/actor/MyActor/enableDirection          false
   /gate/actor/MyActor/useVolumeFrame


This actor records information about particles entering the volume which the actor is attached to. They are two file types for the output: root file (.root) and IAEA file (.IAEAphsp and .IAEAheader). The name of the particle, the kinetic energy, the position along the three axes, the direction along the three axes, the weight are recorded. In a IAEA file, each particle is designated by an integer while the full name of the particle is recorded in the root file. Particles in IAEA files are limited to photons, electrons, positrons, neutrons and protons. The root file has two additional pieces of information: the name of the volume where the particle was produced and the name of the process which produced the particle. It is possible to enable or disable some information in the phase space file::

   /gate/actor/MyActor/enableEkine              false
   /gate/actor/MyActor/enableXPosition          false
   /gate/actor/MyActor/enableYPosition          false
   /gate/actor/MyActor/enableZPosition          false
   /gate/actor/MyActor/enableXDirection         false
   /gate/actor/MyActor/enableYDirection         false
   /gate/actor/MyActor/enableZDirection         false
   /gate/actor/MyActor/enableProductionVolume   false 
   /gate/actor/MyActor/enableProductionProcess  false
   /gate/actor/MyActor/enableParticleName       false
   /gate/actor/MyActor/enableWeight             false
   /gate/actor/MyActor/enableTrackLength        true


By default the frame used for the position and the direction of the particle is the frame of the world. To use the frame of the volume which the actor is attached to, the following command should be used::

   /gate/actor/source/useVolumeFrame

  
By default, the phase space stores particles entering the volume. To store particles exiting the volume, the following command should be used::

   /gate/actor/MyActor/storeOutgoingParticles true

To store all secondary particles created in the volume, use the command::

   /gate/actor/MyActor/storeSecondaries true

Phase spaces built with all secondaries should not be used as source because some particles could be generated several times.

**With ROOT files**, to avoid very big files, it is possible to restrict the maximum size of the phase space. If a phase space reachs the maximum size, the files is closed and a new file is created. The new file has the same name and a suffix is added. The suffix is the number of the file. For instance, instead of one file of 10 GB, user may prefer 10 files of 1 GB. The value of the maximum size is not exactly the size of the file (value is the size of the TTree)::
 
   /gate/actor/MyActor/setMaxFileSize [Value] [Unit (B, kB, MB, GB)]

**The source of the simulation could be a phase space.** Gate read two types of phase space: root files and IAEA phase spaces. Both can be created with Gate. However, Gate could read IAEA phase spaces created with others simulations::

   /gate/source/addSource  [Source name]  phaseSpace

User can add several phase space files. All files should have the same informations about particles. The files are chained::

   /gate/source/[Source name]/addPhaseSpaceFile [File name 1]
   /gate/source/[Source name]/addPhaseSpaceFile [File name 2]

If particles in the phase space are defined in the world frame, user has to used the command::

   /gate/source/[Source name]/setPhaseSpaceInWorldFrame

If the particle type is not defined in the phase space file, user have to give the particle name. It is supposed that all particles have the same name::

   /gate/source/[Source name]/setParticleType [Particle name]

If user have several phase space sources, each source have the same intensity. User can also choose to give at each source an intensity proportionnal to the number of particles in the files attach to the source::

   /gate/source/[Source name]/useNbOfParticleAsIntensity true

For each run, if the number of events is higher than the number of particles in file, each particle is used several times with the same initial conditions. However, it is possible to rotate the particle position and direction around the z axis of the volume (make sure your phase space files have a rotational symmetry). The regular rotation is a rotation with a fixed angle:

:math:`\alpha = \frac{ 2 \pi }{ N_{used} }`

where :math:`N_{used}` is the number of time the particle is used::

   /gate/source/[Source name]/useRegularSymmetry

The random rotation is a rotation with a random angle::

   /gate/source/[Source name]/useRandomSymmetry

By default, all particles in a phase space are used. The particles in the the phase space can be preselected in function of their position in the :math:`(x , y)` plan. For instance, a particle with a origin far from the collimator aperture is not useful and should be ignored. Particles in a :math:`r` cm-radius circle are selected. The particles outside the circle are ignored::

   /gate/source/[Source name]/setRmax [r] [unit]

Thermal Actor
~~~~~~~~~~~~~

This actor records the optical photon deposited energy (photons absorbed by the tissue/material) in the volume which the actor is attached to. It also performs the diffusion of the deposited energy. The output file format is a 3D matrix (voxelised image img/hdr). The Pennes bioheat model is used to describe the diffusion of hear in biological perfused tissues. The Pennes equation is solved analytically via Fourier transformations and convolution theorem. The solution of the diffusion equation is equivalent to convolving the initial conditions (3D energy map) with a Gaussian with a standard deviation :math:`\sigma = \sqrt{2t K_1}`, with t the diffusion time, :math:`K_1` the tissue thermal diffusivity. The blood perfusion term appears in the solution via an exponential function::

   /gate/actor/addActor ThermalActor                 MyActor
   /gate/actor/MyActor/save                          3DMap.hdr
   /gate/actor/MyActor/attachTo                      phantom
   /gate/actor/MyActor/stepHitType                   random
   /gate/actor/MyActor/setPosition                   0. 0. 0. mm
   /gate/actor/MyActor/setVoxelSize                  0.5 0.5 0.5 mm

Tissue thermal property::

   /gate/actor/MyActor/setThermalDiffusivity         0.32 mm2/s

Density and heat capacity should just be in the same unit for both blood and tissue. In the following example, the density is in kg/mm3 and the heat capacity in mJ kg-1 C-1::

   /gate/actor/MyActor/setBloodDensity               1.06E-6
   /gate/actor/MyActor/setBloodHeatCapacity          3.6E6
   /gate/actor/MyActor/setTissueDensity              1.04E-6
   /gate/actor/MyActor/setTissueHeatCapacity         3.65E6
   /gate/actor/MyActor/setBloodPerfusionRate         0.004

During light illumination of a tissue, the thermal heat produced by the optical photons deposited energy does not accumulate locally in the tissue; it diffuses in biological tissues during illumination. This dynamic effect has been taken into account in the GATE code. The n seconds light illumination simulation is sampled into p time frame 3D images by setting the simulation parameter setNumberOfTimeFrames to p. Each of the p sample images is diffused for a duration of [1, 2, ..., p-1] x n/p seconds. The final image illustrating the heat distribution in the tissues at the end of the illumination time is obtained by adding all diffused images to the last n/p seconds illumination image. This thermal energy (or heat) map will continue to diffuse after illumination by setting the parameter setDiffusionTime to the value of interest. At a certain point in time after the initial temperature boost induced by nanoparticles, the temperature of the tissues will go back to its initial value due to diffusion. This boundary condition is taken into account in a post processing-step of the GATE simulation::

   /gate/actor/MyActor/setNumberOfTimeFrames         5
   /gate/actor/MyActor/setDiffusionTime              5 s


Merged Volume Actor
~~~~~~~~~~~~~~~~~~~

Since GATE V8.0, the user has to possibility to add a G4VSolid (or a analytical solid such as: box, cylinder, tessellated, sphere etc...) within a voxellized volume (defined by ImageRegularParametrisedVolume or ImageNestedParametrisedVolume). 

To be done, the user needs an actor and MUST declare the volumes in a specific order. 

Here is a schematic procedure:

1) Declaring a volume containing the voxellized phantom AND the volume(s) to merge with the voxellized phantom
2) Declaring the voxellized phantom
3) Declaring all the analytical solid to add within the voxellized phantom

Here is a simple example::

   # THE CONTAINER VOLUME
   /gate/world/daughters/name GlobalVol
   /gate/world/daughters/insert box
   /gate/GlobalVol/geometry/setXLength 90. mm
   /gate/GlobalVol/geometry/setYLength 90. mm
   /gate/GlobalVol/geometry/setZLength 90. mm
   /gate/GlobalVol/placement/setTranslation 0.0 0.0 0.0 mm
   /gate/GlobalVol/placement/setRotationAxis 1 0 0
   /gate/GlobalVol/placement/setRotationAngle 0 deg
   /gate/GlobalVol/setMaterial Air
   /gate/GlobalVol/vis/setColor cyan
   /gate/GlobalVol/describe

   # THE VOXELLIZED PHANTOM
   /gate/GlobalVol/daughters/name PhantomTest
   /gate/GlobalVol/daughters/insert ImageRegularParametrisedVolume
   /gate/PhantomTest/geometry/setImage phantom_test_without_box.h33
   /gate/PhantomTest/geometry/setRangeToMaterialFile range.dat
   /gate/PhantomTest/placement/setTranslation 0. 0. 0. mm
   /gate/PhantomTest/placement/setRotationAxis 1 0 0
   /gate/PhantomTest/placement/setRotationAngle 0 deg
   /gate/PhantomTest/setSkipEqualMaterials 0 
   /gate/PhantomTest/describe 

   # 2 ANALYTICAL VOLUMES TO MERGE WITHIN VOXELLIZED PHANTOM
   # FIRST VOLUME
   /gate/GlobalVol/daughters/name BoxAir
   /gate/GlobalVol/daughters/insert box
   /gate/BoxAir/geometry/setXLength 10.0 mm/gate/BoxAir/geometry/setYLength 10.0 mm
   /gate/BoxAir/geometry/setZLength 10.0 mm
   /gate/BoxAir/placement/setTranslation -30.0 0.0 0.0 mm
   /gate/BoxAir/placement/setRotationAxis 1 0 0
   /gate/BoxAir/placement/setRotationAngle 0 deg
   /gate/BoxAir/setMaterial Air
   /gate/BoxAir/vis/setColor cyan
   /gate/BoxAir/describe
   # SECOND VOLUME
   /gate/GlobalVol/daughters/name BoxLung
   /gate/GlobalVol/daughters/insert box
   /gate/BoxLung/geometry/setXLength 10.0 mm
   /gate/BoxLung/geometry/setYLength 10.0 mm
   /gate/BoxLung/geometry/setZLength 10.0 mm
   /gate/BoxLung/placement/setTranslation -10.0 0.0 0.0 mm
   /gate/BoxLung/placement/setRotationAxis 1 0 0
   /gate/BoxLung/placement/setRotationAngle 0 deg
   /gate/BoxLung/setMaterial Lung
   /gate/BoxLung/vis/setColor red
   /gate/BoxLung/describe

The final step is to declare the actor. This actor MUST be the first actor declared in the GATE macro. This actor is like a navigator and its influence during the simulation is very important. Here is the declaration of the actor associated to the above example::

   /gate/actor/addActor MergedVolumeActor mergedVol
   /gate/actor/mergedVol/attachTo GlobalVol
   /gate/actor/mergedVol/volumeToMerge BoxAir,BoxLung

For this actor, the order of the declared volume and the declared actor is very important. In the case of dosimetry, the user could add the dosimetry actor (after the MergedVolumeActor) to retrieve the energy deposit in the volume as follows::

   /gate/actor/addActor DoseActor doseMeasurement
   /gate/actor/doseMeasurement/attachTo GlobalVol
   /gate/actor/doseMeasurement/save output/merged_volume.mhd
   /gate/actor/doseMeasurement/stepHitType random
   /gate/actor/doseMeasurement/setPosition 0 0 0 mm
   /gate/actor/doseMeasurement/setVoxelSize 0.5 0.5 0.5 mm
   /gate/actor/doseMeasurement/setSize 90.5 90.5 90.5 mm
   /gate/actor/doseMeasurement/enableEdep true
   /gate/actor/doseMeasurement/enableUncertaintyEdep true
   /gate/actor/doseMeasurement/enableSquaredEdep true
   /gate/actor/doseMeasurement/enableDose false
   /gate/actor/doseMeasurement/enableUncertaintyDose false
   /gate/actor/doseMeasurement/enableSquaredDose false
   /gate/actor/doseMeasurement/enableNumberOfHits true

A complete example is provided here.

Proton Nuclear Information Actor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This actor records information on proton nuclear interactions (number and type). The information can be stored in a phase space file, as illustrated in `imaging/ProtonRadiography <https://github.com/OpenGATE/GateContrib/tree/master/imaging/ProtonRadiography>`_.

.. _filters-label:

MuMapActor
~~~~~~~~~~

In PET recon, it need MuMap to attenuation correction, people can use MuMapActor to get MuMap and sourceMap. Note: voxel Mu Uint(default) is cm-1::

   /gate/actor/addActor MuMapActor getMuMap
   /gate/actor/getMuMap/attachTo world
   /gate/actor/getMuMap/save myMapFileName.mhd
   /gate/actor/getMuMap/setPosition 0 0 0 mm
   /gate/actor/getMuMap/setVoxelSize 2 2 2 mm
   /gate/actor/getMuMap/setResolution 128 128 100
   /gate/actor/getMuMap/setEnergy 511 keV
   /gate/actor/getMuMap/setMuUnit 1 1/mm ##assign Mu uint

Filters
-------

Filters are used to add selectrion criteria on actors. They are also used with reduction variance techniques. They are filters on particle type, particle ID, energy, direction....

All filters listed below can be inverted and generate exact opposite selection::

   /gate/actor/[Actor Name]/[Filter Name]/invert

Filter on particle type
~~~~~~~~~~~~~~~~~~~~~~~

With this filter it is possible to select particle with the name [Particle Name]::

   /gate/actor/[Actor Name]/addFilter                       particleFilter
   /gate/actor/[Actor Name]/particleFilter/addParticle      [Particle Name]

User can select various particles. It is also possible to select particles which has a parent with the name [Particle Name]::

   /gate/actor/[Actor Name]/addFilter                           particleFilter
   /gate/actor/[Actor Name]/particleFilter/addParentParticle    [Particle Name]

For ions, user should use the Geant4 nomenclature (C12[0.0], c11[0.0]...). These names are different from those used for physics. To select all ions except alpha, deuton and triton, there is the key word 'GenericIon'.

It is also possible to filter on the atomic number (Z) and the mass number (A)::

   /gate/actor/[Actor Name]/addFilter                       particleFilter
   /gate/actor/[Actor Name]/particleFilter/addParticleZ      Z
   /gate/actor/[Actor Name]/particleFilter/addParticleA      A

with A and Z being integer values. 

To address all particles with atomic number Z1 OR atomic number Z2, Z3 ...::

   /gate/actor/[Actor Name]/addFilter                       particleFilter
   /gate/actor/[Actor Name]/particleFilter/addParticleZ      Z1
   /gate/actor/[Actor Name]/particleFilter/addParticleZ      Z2
   /gate/actor/[Actor Name]/particleFilter/addParticleZ      Z3

Within atomic number the logical connection on multiple entries is OR, whereas the two types of particle filters, atomic and mass number filter, are connected with logical AND.

To filter on the PDG number of a particle::

   /gate/actor/[Actor Name]/addFilter                       particleFilter
   /gate/actor/[Actor Name]/particleFilter/addParticlePDG      PDG

Hence, there are 3 possibilities to filter (for example) for protons::

   /gate/actor/[Actor Name]/addFilter                       particleFilter
   /gate/actor/[Actor Name]/particleFilter/addParticleZ      1
   /gate/actor/[Actor Name]/particleFilter/addParticleA      1

or::

   /gate/actor/[Actor Name]/addFilter                       particleFilter
   /gate/actor/[Actor Name]/particleFilter/addParticle      proton

or::

   /gate/actor/[Actor Name]/addFilter                       particleFilter
   /gate/actor/[Actor Name]/particleFilter/addParticlePDG      2212


Example: To kill electrons and positrons in the volume MyVolume::

   /gate/actor/addActor     KillActor                    MyActor
   /gate/actor/MyActor/save                              MyOutputFile.txt
   /gate/actor/MyActor/attachTo                          MyVolume
   /gate/actor/MyActor/addFilter                         particleFilter
   /gate/actor/MyActor/particleFilter/addParticle        e-
   /gate/actor/MyActor/particleFilter/addParticle        e+

Filter on particle ID
~~~~~~~~~~~~~~~~~~~~~

In an event, each track has an unique ID. The incident particle has an ID equal to 1. This filter select particles with the ID [Particle ID] or particles which has a parent with the ID [Particle ID]. As for particle filter, user can select many IDs::

   /gate/actor/[Actor Name]/addFilter               IDFilter
   /gate/actor/[Actor Name]/IDFilter/selectID       [Particle ID]

   /gate/actor/[Actor Name]/addFilter                     IDFilter
   /gate/actor/[Actor Name]/IDFilter/selectParentID       [Particle ID]

Example: To kill all particle exept the incident particle in the volume MyVolume (all particles are the children of the incident particle exept the incident particle itself)::

   /gate/actor/addActor    KillActor                   MyActor
   /gate/actor/MyActor/save                            MyOutputFile.txt
   /gate/actor/MyActor/attachTo                        MyVolume
   /gate/actor/MyActor/addFilter                       IDFilter
   /gate/actor/MyActor/IDFilter/selectParentID         1

You cannot combine ID and particleFilter.

Filter on volume
~~~~~~~~~~~~~~~~

This actor is especially useful for reduction variance techniques or for selections on daughter volumes.

Example: To kill particles in volume A and B, children of the volume MyVolume::

   /gate/actor/addActor   KillActor                         MyActor
   /gate/actor/MyActor/save                                 MyOutputFile.txt
   /gate/actor/MyActor/attachTo                             MyVolume
   /gate/actor/MyActor/addFilter                            volumeFilter
   /gate/actor/MyActor/volumeFilter/addVolume               A
   /gate/actor/MyActor/volumeFilter/addVolume               B

Filter on energy
~~~~~~~~~~~~~~~~

This filter allows to select particles with a kinetic energy above a threshold Emin and/or below a threshold Emax::

   /gate/actor/[Actor Name]/addFilter              energyFilter
   /gate/actor/[Actor Name]/energyFilter/setEmin   [Value]  [Unit]
   /gate/actor/[Actor Name]/energyFilter/setEmax   [Value]  [Unit]

Example: To kill particles with an energy above 5 MeV::

   /gate/actor/addActor   KillActor                     MyActor
   /gate/actor/MyActor/save                             MyOutputFile.txt
   /gate/actor/MyActor/attachTo                         MyVolume
   /gate/actor/MyActor/addFilter                        energyFilter
   /gate/actor/MyActor/energyFilter/setEmin             5 MeV

Filter on direction
~~~~~~~~~~~~~~~~~~~

This filter is used to select particle with direction inside a cone centered on the reference axis. The angle between the axis and the edge of the cone is in degree. The axis is defined with the (x,y,z) directions::

   /gate/actor/[Actor Name]/addFilter                    angleFilter
   /gate/actor/[Actor Name]/angleFilter/setAngle         [Value]
   /gate/actor/[Actor Name]/angleFilter/setDirection     [x] [y] [z]

Example: To kill particles in a cone of 20 degrees around x axis::

   /gate/actor/addActor    KillActor                         MyActor
   /gate/actor/MyActor/save                                  MyOutputFile.txt
   /gate/actor/MyActor/attachTo                              MyVolume
   /gate/actor/MyActor/addFilter                             angleFilter
   /gate/actor/MyActor/angleFilter/setAngle                  20
   /gate/actor/MyActor/angleFilter/setDirection              1 0 0

Filter on material type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The attached volume of the actor may contains different materials. This filter is used to select particles hit different materials::

   /gate/actor/[Actor Name]/addFilter                    materialFilter
   /gate/actor/[Actor Name]/materialFilter/addMaterial   [Material Name]

Example: To kill particles in the volume MyVolume that hits Water::

   /gate/actor/addActor    KillActor                         MyActor
   /gate/actor/MyActor/save                                  MyOutputFile.txt
   /gate/actor/MyActor/attachTo                              MyVolume
   /gate/actor/MyActor/addFilter                             materialFilter
   /gate/actor/MyActor/materialFilter/addMaterial            Water
