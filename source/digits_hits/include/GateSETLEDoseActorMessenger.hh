/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \class  GateSETLEDoseActorMessenger
  \author fabien.baldacci@creatis.insa-lyon.fr
	  francois.smekens@creatis.insa-lyon.fr
*/

#ifndef GATESETLEDOSEACTORMESSENGER_HH
#define GATESETLEDOSEACTORMESSENGER_HH

#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "GateUIcmdWithTwoDouble.hh"
#include "GateImageActorMessenger.hh"

class GateSETLEDoseActor;
class GateSETLEDoseActorMessenger : public GateImageActorMessenger
{
public:
  GateSETLEDoseActorMessenger(GateSETLEDoseActor* sensor);
  virtual ~GateSETLEDoseActorMessenger();
  
  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateSETLEDoseActor * pDoseActor;

  G4UIcmdWithABool * pEnableDoseCmd;
  G4UIcmdWithABool * pEnableDoseUncertaintyCmd;
  G4UIcmdWithABool * pEnablePrimaryDoseCmd;
  G4UIcmdWithABool * pEnablePrimaryDoseUncertaintyCmd;
  G4UIcmdWithABool * pEnableSecondaryDoseCmd;
  G4UIcmdWithABool * pEnableSecondaryDoseUncertaintyCmd;

  G4UIcmdWithABool * pEnableHybridinoCmd;
  G4UIcmdWithAnInteger * pSetPrimaryMultiplicityCmd;
  G4UIcmdWithAnInteger * pSetSecondaryMultiplicityCmd;
  GateUIcmdWithTwoDouble *pSetSecondaryMultiplicityCmd2;
};

#endif /* end #define GATESETLEDOSEACTORMESSENGER_HH*/
