/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class GateKillActor
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
 */

#include "GateConfiguration.h"
#ifndef GATEKILLACTOR_HH
#define GATEKILLACTOR_HH

#include "GateVActor.hh"
#include "GateActorMessenger.hh"

//-----------------------------------------------------------------------------
class GateKillActor : public GateVActor
{
 public:

  virtual ~GateKillActor();

  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateKillActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  //-----------------------------------------------------------------------------
  // Callbacks
  virtual G4bool ProcessHits(G4Step * step , G4TouchableHistory* th);
  virtual void clear(){ResetData();}
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}
  //-----------------------------------------------------------------------------
  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

protected:
  GateKillActor(G4String name, G4int depth=5);

  long int mNumberOfTrack;

  GateActorMessenger* pMessenger;
};

MAKE_AUTO_CREATOR_ACTOR(KillActor,GateKillActor)


#endif /* end #define GATEKILLACTOR_HH */
