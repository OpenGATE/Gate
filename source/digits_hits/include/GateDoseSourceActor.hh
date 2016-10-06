/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEDOSESOURCEACTOR_HH
#define GATEDOSESOURCEACTOR_HH

#include "GateConfiguration.h"
#include "GateVImageActor.hh"
#include "GateActorMessenger.hh"
#include "GateDoseSourceActorMessenger.hh"
#include "GateImageOfHistograms.hh"
#include "GateSourceMgr.hh"


//-----------------------------------------------------------------------------
class GateDoseSourceActor: public GateVImageActor
{
public:
  virtual ~GateDoseSourceActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateDoseSourceActor)

  virtual void Construct();
  virtual void UserPreTrackActionInVoxel(const int index, const G4Track* t);
  virtual void UserPostTrackActionInVoxel(const int index, const G4Track* t);
  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step);
  virtual void BeginOfEventAction(const G4Event * e);
  //virtual void BeginOfRunAction(const G4Event * e);

  virtual void SaveData();
  virtual void ResetData();

  void SetSpotIDFromSource(G4String nameOfSource){bSourceName = nameOfSource;SpotOrNot=true;}
  void SetLayerIDFromSource(G4String nameOfSource){bSourceName = nameOfSource;SpotOrNot=false;}

protected:
  GateDoseSourceActor(G4String name, G4int depth=0);
  GateDoseSourceActorMessenger * pMessenger;

  GateSourceTPSPencilBeam *tpspencilsource;
  G4String bSourceName;
  GateImageOfHistograms * doseSourceImage;  //main output (yield)
  int bID; //current spotid OR layerID set at beginofeventaction
  bool areweinityet;
  bool SpotOrNot;

};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(DoseSourceActor,GateDoseSourceActor)

#endif // end GATEDOSESOURCEACTOR
