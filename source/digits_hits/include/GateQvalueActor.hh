/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*!
  \class  GateSecondaryProductionActor
  \author thibault.frisson@creatis.insa-lyon.fr
  laurent.guigues@creatis.insa-lyon.fr
  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEQVALUEACTOR_HH
#define GATEQVALUEACTOR_HH

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "GateImageActorMessenger.hh"
#include "GateImageWithStatistic.hh"

#include "G4UnitsTable.hh"

class GateQvalueActor : public GateVImageActor
{
public:

  //-----------------------------------------------------------------------------
  // Actor name

  virtual ~GateQvalueActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateQvalueActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void BeginOfEventAction(const G4Event * event);
  virtual void EndOfEventAction(const G4Event * event);

  virtual void UserSteppingActionInVoxel(const int /*index*/, const G4Step* /*step*/);
  virtual void UserPreTrackActionInVoxel(const  int index, const G4Track* t) ;
  virtual void UserPostTrackActionInVoxel(const int index, const G4Track* t) ;

  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}

protected:
  GateQvalueActor(G4String name, G4int depth=0);
  GateImageActorMessenger * pMessenger;

  int mCurrentEvent;
  G4String mQvalueFilename;
  GateImageWithStatistic mQvalueImage;
  std::map<G4String,G4double> listOfEmiss;
  int mNSec;
};

MAKE_AUTO_CREATOR_ACTOR(QvalueActor,GateQvalueActor)

#endif
