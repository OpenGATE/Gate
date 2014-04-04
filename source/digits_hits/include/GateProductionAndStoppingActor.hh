/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*!
  \class  GateProductionAndStoppingActor
  \author thibault.frisson@creatis.insa-lyon.fr
  laurent.guigues@creatis.insa-lyon.fr
  david.sarrut@creatis.insa-lyon.fr

  modified by I. Martinez-Rovira (immamartinez@gmail.com)
*/

#ifndef GATEPRODANDSTOPACTOR_HH
#define GATEPRODANDSTOPACTOR_HH

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "GateImageActorMessenger.hh"
#include "GateImageWithStatistic.hh"

#include "G4UnitsTable.hh"

class GateProductionAndStoppingActor : public GateVImageActor
{
public:

  //-----------------------------------------------------------------------------
  // Actor name

  virtual ~GateProductionAndStoppingActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateProductionAndStoppingActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void BeginOfEventAction(const G4Event * event);

  virtual void UserPreTrackActionInVoxel(const  int /*index*/, const G4Track* /*t*/){}
  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step) ;
  virtual void UserPostTrackActionInVoxel(const int index, const G4Track* t) ;

  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}


protected:
  GateProductionAndStoppingActor(G4String name, G4int depth=0);
  GateImageActorMessenger * pMessenger;

  int mCurrentEvent;

  GateImageWithStatistic mProdImage;
  GateImageWithStatistic mStopImage;

  G4String mProdFilename;
  G4String mStopFilename;

};

MAKE_AUTO_CREATOR_ACTOR(ProductionAndStoppingActor,GateProductionAndStoppingActor)

#endif /* end #define GATESIMULATIONSTATISTICACTOR_HH */
