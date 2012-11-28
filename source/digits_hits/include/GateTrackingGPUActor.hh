/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  GateTrackingGPUActor
  Track particles in voxelized volume with GPU
  July 2012
*/

#ifndef GATETRACKINGGPUACTOR_HH
#define GATETRACKINGGPUACTOR_HH

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "G4UnitsTable.hh"
#include "GateTrackingGPUActorMessenger.hh"
#include "GateImageWithStatistic.hh"
#include "GateGPUIO.hh"

class GateTrackingGPUActor: public GateVActor
{
 public: 
  
  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateTrackingGPUActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateTrackingGPUActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  //-----------------------------------------------------------------------------
  virtual void BeginOfRunAction(const G4Run*r);
  // virtual void BeginOfEventAction(const G4Event * event);
  // virtual void PreUserTrackingAction(const GateVVolume * v, const G4Track*t);
  // virtual void PostUserTrackingAction(const GateVVolume * v, const G4Track*t);
  virtual void UserSteppingAction(const GateVVolume * v, const G4Step*);

  //-----------------------------------------------------------------------------
  virtual void SaveData();
  virtual void ResetData();  

  void SetGPUDeviceID(int n);
  void SetGPUBufferSize(int n);

  //-----------------------------------------------------------------------------
protected:
  GateTrackingGPUActor(G4String name, G4int depth=0);
  GateTrackingGPUActorMessenger * pMessenger;

  void CreateNewParticle(const GateGPUIO_Particle & p);

  // Input and output structure for gpu.
  GateGPUIO_Input * gpu_input;
  GateGPUIO_Output * gpu_output;
  
  // max buffer size
  unsigned int max_buffer_size;
  int mGPUDeviceID;
};

MAKE_AUTO_CREATOR_ACTOR(TrackingGPUActor,GateTrackingGPUActor)

#endif /* end #define GATESIMULATIONSTATISTICACTOR_HH */
