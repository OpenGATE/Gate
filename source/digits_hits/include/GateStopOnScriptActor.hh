/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class GateStopOnScriptActor
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
 */

#ifndef GATESTOPONSCRIPTACTOR_HH
#define GATESTOPONSCRIPTACTOR_HH

#include "GateVActor.hh"
#include "GateActorManager.hh"
#include "GateStopOnScriptActorMessenger.hh"

//-----------------------------------------------------------------------------
/// \brief Actor displaying nb events/tracks/step
class GateStopOnScriptActor : public GateVActor
{
 public:

  virtual ~GateStopOnScriptActor();

  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateStopOnScriptActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  //-----------------------------------------------------------------------------
  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  void EnableSaveAllActors(bool b);

protected:
  GateStopOnScriptActor(G4String name, G4int depth=0);
  GateStopOnScriptActorMessenger * pMessenger;
  bool mSaveAllActors;
};

MAKE_AUTO_CREATOR_ACTOR(StopOnScriptActor,GateStopOnScriptActor)


#endif /* end #define GATESTOPONSCRIPTACTOR_HH */
