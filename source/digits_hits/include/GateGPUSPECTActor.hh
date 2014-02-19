/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  GateGPUSPECTActor
  Track particles in collimator with GPU
  September 2013
*/

#ifndef GATEGPUSPECTACTOR_HH
#define GATEGPUSPECTACTOR_HH

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "G4UnitsTable.hh"
#include "GateGPUSPECTActorMessenger.hh"
#include "GateImageWithStatistic.hh"
#include "GateGPUCollimIO.hh"

class GateGPUSPECTActor: public GateVActor
{
 public: 
  
  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateGPUSPECTActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateGPUSPECTActor)

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
  virtual void SaveData();
  virtual void ResetData();  

  void SetGPUDeviceID(G4int n);
  void SetGPUBufferSize(G4int n);
  void SetHoleHexaHeight(G4double d);
  void SetHoleHexaRadius(G4double d);
  void SetHoleHexaRotAxis(G4ThreeVector v);
  void SetHoleHexaRotAngle(G4double d);
  void SetHoleHexaMaterial(G4String m);
  void SetCubArrayRepNumX(G4int n);
  void SetCubArrayRepNumY(G4int n);
  void SetCubArrayRepNumZ(G4int n);
  void SetCubArrayRepVec(G4ThreeVector v);
  void SetLinearRepNum(G4int n);
  void SetLinearRepVec(G4ThreeVector v);

  //-----------------------------------------------------------------------------
protected:
  GateGPUSPECTActor(G4String name, G4int depth=0);
  GateGPUSPECTActorMessenger * pMessenger;

  // Input structure for gpu.
  GateGPUCollimIO_Input * gpu_input;
 
  // Init GPU
  Materials gpu_materials;
  Colli gpu_collim;
  StackParticle gpu_photons, cpu_photons;
  CoordHex2 gpu_centerOfHexagons, cpu_centerOfHexagons;
  //CoordHex2 gpu_sixCorners;

  // Half phantom size
  //float half_phan_size_x, half_phan_size_y, half_phan_size_z;

  // max buffer size
  unsigned int max_buffer_size;
  unsigned int ct_photons;
  G4int mGPUDeviceID;
 
  // collimator features (hexagonal hole)
  G4double mHoleHexaHeight;
  G4double mHoleHexaRadius;
  G4ThreeVector mHoleHexaRotAxis;
  G4double mHoleHexaRotAngle;
  G4String mHoleHexaMat;
  G4int mCubArrayRepNumX;
  G4int mCubArrayRepNumY;
  G4int mCubArrayRepNumZ;
  G4double mCubArrayRepVecX;
  G4double mCubArrayRepVecY;
  G4double mCubArrayRepVecZ;
  G4int mLinearRepNum;
  G4double mLinearRepVecX;
  G4double mLinearRepVecY;
  G4double mLinearRepVecZ;
  
};

MAKE_AUTO_CREATOR_ACTOR(GPUSPECTActor,GateGPUSPECTActor)

#endif /* end #define GATEGPUSPECTACTOR_HH */
