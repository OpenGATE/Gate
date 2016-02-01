/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateLETActorMessenger
*/

#ifndef GATELETACTORMESSENGER_HH
#define GATELETACTORMESSENGER_HH

#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "GateImageActorMessenger.hh"
#include "G4SystemOfUnits.hh" 

class GateLETActor;
class GateLETActorMessenger : public GateImageActorMessenger
{
public:
  GateLETActorMessenger(GateLETActor* sensor);
  virtual ~GateLETActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);
  
protected:
  GateLETActor * pLETActor;

  G4UIcmdWithABool * pSetRestrictedCmd;
  G4UIcmdWithADoubleAndUnit * pSetDeltaRestrictedCmd;
  
  G4UIcmdWithABool * pEnableLETUncertaintyCmd;
  G4UIcmdWithABool * pSetDoseToWaterCmd;
  G4UIcmdWithAString * pAveragingTypeCmd; 
};

#endif /* end #define GATELETACTORMESSENGER_HH*/
