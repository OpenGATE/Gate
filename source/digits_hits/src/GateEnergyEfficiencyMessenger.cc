/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateEnergyEfficiencyMessenger.hh"
#include "GateEnergyEfficiency.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "GateVDistribution.hh"
#include "GateDistributionListManager.hh"

GateEnergyEfficiencyMessenger::GateEnergyEfficiencyMessenger(GateEnergyEfficiency* itsPulseProcessor)
    : GatePulseProcessorMessenger(itsPulseProcessor)
{
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName() + "setDistribution";
  distNameCmd = new G4UIcmdWithAString(cmdName,this);
  distNameCmd->SetGuidance("Set the efficiency distribution");

}


GateEnergyEfficiencyMessenger::~GateEnergyEfficiencyMessenger()
{
  delete distNameCmd;
}


void GateEnergyEfficiencyMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  GateEnergyEfficiency* localEff = (GateEnergyEfficiency*)GetNamedObject();
  if (!localEff) return;
  if ( command==distNameCmd ){
    GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
    if (distrib) localEff->SetEfficiency(distrib);}
  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
