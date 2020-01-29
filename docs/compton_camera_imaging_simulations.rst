.. _compton_camera_imaging_simulations-label:

Compton camera imaging simulations
=======================================

.. contents:: Table of Contents
   :depth: 15
   :local:

Introduction
------------

The Compton camera imaging system has been designed as an actor (see  :ref:`tools_to_interact_with_the_simulation_actors-label`) that collects the information of the hits in the different layers of the system. The following commands must be employed to add and attach the actor to a volume that contains the whole system::

	/gate/actor/addActor  ComptonCameraActor      [Actor Name]
	/gate/actor/[Actor Name]/attachTo             [Vol Name]            

A Compton camera system is typically composed of two types of detectors: the scatterer and the absorber. These terms work as key words within the actor. The behavior of the  volumes associated to absorber and scatterer sensitive detectors (SD) is equivalent to the crystalSD  behavior  in  PET/SPECT systems. The sensitive detectors are specified with the following commands::

	/gate/actor/[Actor Name]/absorberSDVolume      [absorber Name]
	/gate/actor/[Actor Name]/scattererSDVolume     [scatterer Name]


