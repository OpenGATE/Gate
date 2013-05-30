/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateScintillatorResponseActor
  \author simon.rit@creatis.insa-lyon.fr
 */

#include "GateConfiguration.h"
#ifdef GATE_USE_RTK

#ifndef GATESCINTILLATORRESPONSEACTOR_HH
#define GATESCINTILLATORRESPONSEACTOR_HH

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "GateMiscFunctions.hh"
#include "GateScintillatorResponseActorMessenger.hh"

class GateScintillatorResponseActor : public GateVImageActor
{
 public: 
  
  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateScintillatorResponseActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateScintillatorResponseActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  void EnableScatterImage(bool b) { mIsScatterImageEnabled = b; }

  virtual void BeginOfEventAction(const G4Event * e);
  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step);
  virtual void UserPreTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}
  virtual void UserPostTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}

 /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();
  void ReadMuAbsortionList(G4String filename);

  ///Scorer related
  //virtual G4bool ProcessHits(G4Step *, G4TouchableHistory*);
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}
  void SetScatterOrderFilename(G4String name) { mScatterOrderFilename = name; }

protected:
  GateScintillatorResponseActor(G4String name, G4int depth=0);
  GateScintillatorResponseActorMessenger * pMessenger;

  G4AffineTransform mDetectorToWorld;
  std::map< G4double, G4double > mUserMuAbsortionMap;
  bool mIsScatterImageEnabled;
  GateImage mImageScatter;
  std::vector<GateImage *> mScintillatorResponsePerOrderImages;
  G4String mScatterOrderFilename;
};

MAKE_AUTO_CREATOR_ACTOR(ScintillatorResponseActor,GateScintillatorResponseActor)

#endif /* end #define GATESIMULATIONSTATISTICACTOR_HH */

#endif // GATE_USE_RTK
