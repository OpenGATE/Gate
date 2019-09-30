Cut and Variance Reduction Technics
===================================

.. contents:: Table of Contents
   :depth: 15
   :local:

Production threshold, step limiter and special cuts
---------------------------------------------------

Production threshold
~~~~~~~~~~~~~~~~~~~~

To avoid infrared divergence, charged particles processes (ionization and bremsstrahlung) require a threshold below which no secondary particles will be generated. Because of this requirement, gammas, electrons and positrons require production thresholds which the user should define. This threshold should be defined as a distance, or range cut-off, which is internally converted to an energy for individual materials. Production thresholds are defined for a geometrical region. In GATE, each volume is considered as a geometrical region. If no cut is defined, the region inherited the threshold of the parent volume. The default cut value of the world is set to 1.0 mm::

   /gate/physics/Gamma/SetCutInRegion      [Volume Name]  [Cut value] [Unit] 
   /gate/physics/Electron/SetCutInRegion   [Volume Name]  [Cut value] [Unit]
   /gate/physics/Positron/SetCutInRegion   [Volume Name]  [Cut value] [Unit]

For example::

   /gate/physics/Gamma/SetCutInRegion     world  1.0 mm

The list of production threshold in range and in energy for each volume can be display with the command::

   /gate/physics/displayCuts

User should use this command after the initilization.

X-Ray cuts and auger electrons in photo-electric process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Secondary electron and photon production in low energy photo-electric process (livermore and penelope models) could be customized.

The auger electron could be activated in livermore and penelope models::

   /gate/physics/processes/PhotoElectric/setAugerElectron true

In livermore model, an energy threshold could be defined from which secondary particles are produced. The threshold is defined for the whole world of the simulation::

   /gate/physics/processes/PhotoElectric/setDeltaRayCut [value] [unit]
   /gate/physics/processes/PhotoElectric/setXRayCut [value] [unit]


Step limiter
~~~~~~~~~~~~

**< ! > This part is recommended for advanced users only!**

The step limiter can be considered as a process with fixed step size. It allows to limit the maximum size of step. As for production threshold, the step limiter is defined for a geometrical region and the region can inherit the step limiter of the parent volume. User have to define the step size in the region::

   /gate/physics/SetMaxStepSizeInRegion [Volume Name] [Step size] [Unit]

For example::

   /gate/physics/SetMaxStepSizeInRegion world  1.0 mm

Then, the step limiter process has to be add to particles::

   /gate/physics/ActivateStepLimiter proton

Special cuts
~~~~~~~~~~~~

**< ! > This part is recommended for advanced users only!**

The user can define four cuts to limit the tracking of a particle:

* the maximum total track length
* the maximum total time of flight
* the minimum kinetic energy
* the minimum remaining range 

While step limiter is affected to a step, special cuts are affected to a track. When a particle is stopped, the energy is deposited locally. As for production threshold, the special cuts are defined for a geometrical region and the region can inherit the special cuts of the parent volume.

User have to define the values of the special cuts in the region and activate the cuts for particles:: 

   /gate/physics/SetMaxToFInRegion world 5 s
   /gate/physics/SetMinKineticEnergyInRegion world 1 keV
   /gate/physics/SetMaxTrackLengthInRegion world 0.01 mm
   /gate/physics/SetMinRemainingRangeInRegion world 0.02 mm
   /gate/physics/ActivateSpecialCuts proton 

The user does not have to define all the cuts. The *ActivateSpecialCuts* command is effective for all the special cuts that are defined.

Variance reduction
------------------

Two standard reduction variance techniques are available in GATE: splitting and russian roulette. The weight of secondary particles is recalculated in function of the number of secondaries generated. User can also defined filters to increase the efficiency of these techniques. 

**< ! > User have to verify that all the tools he used in his simulation take into account particle weight!**


Splitting and russian roulette
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Splitting
^^^^^^^^^

In this technique, the final state of the process is generated N times and the weight of each secondary is 1/N::

   /gate/physics/processes/Bremsstrahlung/activateSplitting [Particle] [N]
   
   Parameter : [Particle]
   Parameter type  : s
   Omittable       : False
   
   Parameter : [N]
   Parameter type  : i
   Omittable       : False

Example: to split 100 times the electron bremsstrahlung photon (not that we specify that the e- is the particle which do the bremsstrahlung, but the split is applied on the generated photon)::

   /gate/physics/processes/Bremsstrahlung/activateSplitting e- 100

Russian roulette
^^^^^^^^^^^^^^^^

In this technique, Russian roulette is played on secondary particles. The survival probability is 1/N and the weight of each secondary is N::

   /gate/physics/processes/Bremsstrahlung/activateRussianRoulette [Particle] [N]
   
   Parameter : [Particle]
   Parameter type  : s
   Omittable       : False
   
   Parameter : [N]
   Parameter type  : i
   Omittable       : False

Example: to keep 2% of electron bremsstrahlung photon (2/100 = 1/50)::

   /gate/physics/processes/Bremsstrahlung/activateRussianRoulette e- 50

Selective splitting and russian roulette
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To increase the efficiency of the splitting and the russian roulette technique, user can add selections criteria on the incident (primary) or secondary particles. The selection is done with filters. The filters for splitting and russian roulette are the same as for Actors. For example, to split bremsstrahlung photons with a vector direction inside a cone of 20 degrees around the x axis::

   /gate/physics/processes/Bremsstrahlung/addFilter angleFilter secondaries
   /gate/physics/processes/Bremsstrahlung/secondaries/angleFilter/setAngle 20
   /gate/physics/processes/Bremsstrahlung/secondaries/angleFilter/setDirection 1 0 0

There are several filters types: filters on particle, particle ID, energy, direction, volume... See the chapter on Actor for a description of all filters.

TLE and seTLE (Track Length Estimator)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the TLEDoseActor and SETLEDoseActor :ref:`tle_and_setle_track_length_estimator-label`.
