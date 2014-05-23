/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEEmCalculatorActorMESSENGER_CC
#define GATEEmCalculatorActorMESSENGER_CC

#include "GateEmCalculatorActorMessenger.hh"
#include "GateEmCalculatorActor.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"



//-----------------------------------------------------------------------------
GateEmCalculatorActorMessenger::GateEmCalculatorActorMessenger(GateEmCalculatorActor* sensor):GateActorMessenger(sensor),pEmCalculatorActor(sensor)
{
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateEmCalculatorActorMessenger::~GateEmCalculatorActorMessenger()
{
  delete pSetEnergyCmd;
  delete pSetParticleNameCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEmCalculatorActorMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;

  bb = base+"/setEnergy";
  pSetEnergyCmd = new G4UIcmdWithADoubleAndUnit(bb,this);
  guidance = "Set the energy for the calculation.";
  pSetEnergyCmd->SetGuidance(guidance);

  bb = base+"/setParticleName";
  pSetParticleNameCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Set the particle name for the calculation.";
  pSetParticleNameCmd->SetGuidance(guidance);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEmCalculatorActorMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command == pSetEnergyCmd) pEmCalculatorActor->SetEnergy(pSetEnergyCmd->GetNewDoubleValue(param));
  if(command == pSetParticleNameCmd) pEmCalculatorActor->SetParticleName(param);

  GateActorMessenger::SetNewValue(command ,param );
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEEmCalculatorActorMESSENGER_CC */
