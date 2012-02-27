/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/
#ifdef G4ANALYSIS_USE_ROOT

/*
  \class  GateAugerDetectorActorMessenger
  \author pierre.gueth@creatis.insa-lyon.fr
*/

#ifndef GATEAUGERDETECTORACTORMESSENGER_HH
#define GATEAUGERDETECTORACTORMESSENGER_HH

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

#include "GateActorMessenger.hh"

class GateAugerDetectorActor;

//-----------------------------------------------------------------------------
class GateAugerDetectorActorMessenger : public GateActorMessenger 
{
 public: 
  
  GateAugerDetectorActorMessenger(GateAugerDetectorActor * v);
  virtual ~GateAugerDetectorActorMessenger();
  virtual void SetNewValue(G4UIcommand*, G4String);

protected:
  void BuildCommands(G4String base);

  /// Associated sensor
  GateAugerDetectorActor * pActor; 

  /// Command objects
  //G4UIcmdWithAnInteger * pNBinsCmd;
  G4UIcmdWithADoubleAndUnit* pMaxTOFCmd;
  G4UIcmdWithADoubleAndUnit* pMinEdepCmd;

}; // end class GateAugerDetectorActorMessenger
//-----------------------------------------------------------------------------

#endif /* end #define GATEAUGERDETECTORACTORMESSENGER_HH */
#endif
