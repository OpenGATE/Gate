/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  GateGPUPhotRadTheraActor
  Feb 2013
*/

#ifndef GateGPUPhotRadTheraActor_HH
#define GateGPUPhotRadTheraActor_HH

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "G4UnitsTable.hh"
#include "GateGPUPhotRadTheraActorMessenger.hh"
#include "GateImageWithStatistic.hh"
#include "GateGPUIO.hh"

class GateGPUPhotRadTheraActor: public GateVActor
{
 public:

  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateGPUPhotRadTheraActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateGPUPhotRadTheraActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  //-----------------------------------------------------------------------------
  virtual void BeginOfRunAction(const G4Run*r);
  virtual void EndOfRunAction(const G4Run *r);
  // virtual void BeginOfEventAction(const G4Event * event);
  // virtual void PreUserTrackingAction(const GateVVolume * v, const G4Track*t);
  // virtual void PostUserTrackingAction(const GateVVolume * v, const G4Track*t);
  virtual void UserSteppingAction(const GateVVolume * v, const G4Step*);

  //-----------------------------------------------------------------------------
  virtual void SaveData(){};
  virtual void ResetData();

  void SetGPUDeviceID(int n);
  void SetGPUBufferSize(int n);

  //-----------------------------------------------------------------------------
protected:
  GateGPUPhotRadTheraActor(G4String name, G4int depth=0);
  GateGPUPhotRadTheraActorMessenger * pMessenger;

  GateGPUIO_Input * gpu_input;

  // Init GPU
  Dosimetry gpu_dosemap;
  Materials gpu_materials;
  Volume gpu_phantom;
  StackParticle gpu_photons, gpu_electrons;
  StackParticle cpu_photons;

  // Half phantom size
  float half_phan_size_x, half_phan_size_y, half_phan_size_z;

  // max buffer size
  unsigned int max_buffer_size;
  unsigned int ct_photons;
  int mGPUDeviceID;
};

MAKE_AUTO_CREATOR_ACTOR(GPUPhotRadTheraActor,GateGPUPhotRadTheraActor)

#endif /* end #define GateGPUPhotRadTheraActor_HH */
