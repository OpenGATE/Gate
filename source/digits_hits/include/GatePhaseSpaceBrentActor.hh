/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GatePhaseSpaceBrentActor
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr

	  DoseToWater option added by Loïc Grevillot
  \date	March 2011
 */

#ifndef GatePhaseSpaceBrentACTOR_HH
#define GatePhaseSpaceBrentACTOR_HH

#include <G4NistManager.hh>

#include "GateActorManager.hh"
#include "GatePhaseSpaceActor.hh"
#include "GatePhaseSpaceBrentActorMessenger.hh"

class GatePhaseSpaceBrentActor : public GatePhaseSpaceActor
{
 public:

  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GatePhaseSpaceBrentActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GatePhaseSpaceBrentActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);
  virtual void BeginOfEventAction(const G4Event * event);

  virtual void Initialize(G4HCofThisEvent*){};
  virtual void EndOfEvent(G4HCofThisEvent*){};


  virtual void SaveData();
  virtual void ResetData();

protected:
  GatePhaseSpaceBrentActor(G4String name, G4int depth=0);
  GatePhaseSpaceBrentActorMessenger * pMessenger;
  float primaryEnergy;

};

MAKE_AUTO_CREATOR_ACTOR(BrentActor,GatePhaseSpaceBrentActor)

#endif /* end #define GATESIMULATIONSTATISTICACTOR_HH */
