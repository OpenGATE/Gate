How to use Gate on a GPU
========================

.. contents:: Table of Contents
   :depth: 15

Installation of CUDA tools
--------------------------

IMPORTANT NOTE: 

**!! GPU modules can be used ONLY with NVIDIA hardware and CUDA tools !!**

For details, see the link on the wiki installation guide page:
http://wiki.opengatecollaboration.org/index.php/New_Compilation_ProcedureV7.0#GPU_.26_CUDA_tools

PET applications
----------------

For PET applications, the GPU manages the particle tracking within the voxelized phantom and this is why the source type is defined as a [GPUEmisTomo].
Examples are provided within the GATE source package and users can find a complete example about How To define and run a complete PET simulation set-up by using a GPU architecture.

IMPORTANT NOTES: 

1) By using the GPU, phantom scatter informations are NOT available in the output files
2) The sources radiactive decay time is NOT available with the GPU

Example::

   #==================================================
   # VOXELIZED SOURCES
   #==================================================
   /gate/source/addSource                                              srcvoxel GPUEmisTomo
   /gate/source/srcvoxel/attachPhantomTo ncat
   /gate/source/srcvoxel/setGPUBufferSize                              1000000
   /gate/source/srcvoxel/setGPUDeviceID                                1
   # Read the phantom as usual
   /gate/source/srcvoxel/reader/insert                                 interfile
   /gate/source/srcvoxel/interfileReader/translator/insert             range
   /gate/source/srcvoxel/interfileReader/rangeTranslator/readTable     data/activities.dat
   /gate/source/srcvoxel/interfileReader/rangeTranslator/describe      1
   /gate/source/srcvoxel/interfileReader/verbose                       0
   /gate/source/srcvoxel/interfileReader/readFile                      data/thorax_phantom.hdr
   /gate/source/srcvoxel/setType                                       backtoback
   /gate/source/srcvoxel/gps/particle                                  gamma
   /gate/source/srcvoxel/gps/energytype                                Mono
   /gate/source/srcvoxel/gps/monoenergy                                0.511 MeV
   /gate/source/srcvoxel/setPosition                                   0 0 0 cm
   /gate/source/srcvoxel/gps/confine                                   NULL
   /gate/source/srcvoxel/gps/angtype                                   iso



CT applications
---------------

For CT applications, the GPU tracking is defined as an Actor.

Examples are provided within the GATE source package and users can find a complete example about How To define and run a complete CT simulation set-up by using a GPU architecture. 

IMPORTANT NOTE: 

The scannerCT system, the digitizer and the CT deticated output are NOT available with the GPU CT module. The CT detector must be define by using an actor as it is shown in the complete example provided with the source::

   #==================================================
   # GPU Tracking
   #==================================================
   /gate/actor/addActor                  GPUTransTomoActor gpuactor
   /gate/actor/gpuactor/attachTo         patient
   /gate/actor/gpuactor/setGPUDeviceID   1
   /gate/actor/gpuactor/setGPUBufferSize 10000000   # 1M buffer size =  400MB on the GPU
                                                    # 5M             =  760MB
                                                    #10M             = 1300MB

Optical applications
--------------------

For optical applications, the GPU manages the particle tracking within the voxelized phantom and this is why the source type is defined as a [GPUOpticalVoxel]. Examples are provided within the GATE source package and users can find a complete example about How To define and run a complete optical simulation set-up by using a GPU architecture::

   #==================================================
   # GPU Tracking for optical applications
   # V O X E L    S O U R C E
   # GPU VERSION
   #==================================================
    /gate/source/addSource		        voxel   GPUOpticalVoxel
    /gate/source/voxel/attachPhantomTo 		biolumi
    /gate/source/voxel/setGPUBufferSize 	        5000000
    /gate/source/voxel/setGPUDeviceID 		1
    /gate/source/voxel/energy 			6.0 eV

*last modification: 11/04/2019*
