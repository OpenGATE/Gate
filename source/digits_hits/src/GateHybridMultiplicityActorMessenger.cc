/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEHYBRIDMULTIPLICITYACTORMESSENGER_CC
#define GATEHYBRIDMULTIPLICITYACTORMESSENGER_CC

#include "GateHybridMultiplicityActorMessenger.hh"
#include "GateHybridMultiplicityActor.hh"

//-----------------------------------------------------------------------------
GateHybridMultiplicityActorMessenger::GateHybridMultiplicityActorMessenger(GateHybridMultiplicityActor* sensor)
  :GateActorMessenger(sensor),
   pMultiplicityActor(sensor)
{
  pSetPrimaryMultiplicityCmd = 0;
  pSetSecondaryMultiplicityCmd = 0;
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateHybridMultiplicityActorMessenger::~GateHybridMultiplicityActorMessenger()
{
  if(pSetPrimaryMultiplicityCmd) delete pSetPrimaryMultiplicityCmd;
  if(pSetSecondaryMultiplicityCmd) delete pSetSecondaryMultiplicityCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHybridMultiplicityActorMessenger::BuildCommands(G4String base)
{
  G4String n = base+"/setPrimaryMultiplicity";
  pSetPrimaryMultiplicityCmd = new G4UIcmdWithAnInteger(n, this); 
  G4String guid = G4String("Set the number of hybrid particle by primary particle generated");
  pSetPrimaryMultiplicityCmd->SetGuidance(guid);

  n = base+"/setSecondaryMultiplicity";
  pSetSecondaryMultiplicityCmd = new G4UIcmdWithAnInteger(n, this); 
  guid = G4String("Set the number of hybrid particle by secondary particle generated");
  pSetSecondaryMultiplicityCmd->SetGuidance(guid);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHybridMultiplicityActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  GateActorMessenger::SetNewValue(cmd, newValue);
  if (cmd == pSetPrimaryMultiplicityCmd) pMultiplicityActor->SetPrimaryMultiplicity(pSetPrimaryMultiplicityCmd->GetNewIntValue(newValue));
  if (cmd == pSetSecondaryMultiplicityCmd) pMultiplicityActor->SetSecondaryMultiplicity(pSetSecondaryMultiplicityCmd->GetNewIntValue(newValue));
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEHYBRIDDOSEACTORMESSENGER_CC */
