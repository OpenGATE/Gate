/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateHybridDoseActorMessenger
  \author fabien.baldacci@creatis.insa-lyon.fr
	  francois.smekens@creatis.insa-lyon.fr
*/

#ifndef GATEHYBRIDDOSEACTORMESSENGER_HH
#define GATEHYBRIDDOSEACTORMESSENGER_HH

#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "GateUIcmdWithTwoDouble.hh"
#include "GateImageActorMessenger.hh"

class GateHybridDoseActor;
class GateHybridDoseActorMessenger : public GateImageActorMessenger
{
public:
  GateHybridDoseActorMessenger(GateHybridDoseActor* sensor);
  virtual ~GateHybridDoseActorMessenger();
  
  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateHybridDoseActor * pDoseActor;

  G4UIcmdWithABool * pEnableDoseCmd;
  G4UIcmdWithABool * pEnableDoseUncertaintyCmd;
  G4UIcmdWithABool * pEnablePrimaryDoseCmd;
  G4UIcmdWithABool * pEnablePrimaryDoseUncertaintyCmd;
  G4UIcmdWithABool * pEnableSecondaryDoseCmd;
  G4UIcmdWithABool * pEnableSecondaryDoseUncertaintyCmd;

  G4UIcmdWithAnInteger * pSetPrimaryMultiplicityCmd;
  G4UIcmdWithAnInteger * pSetSecondaryMultiplicityCmd;
  GateUIcmdWithTwoDouble *pSetSecondaryMultiplicityCmd2;
};

#endif /* end #define GATEHYBRIDDOSEACTORMESSENGER_HH*/
