/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateLocalEfficiencyMessenger.hh"
#include "GateLocalEfficiency.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "GateVDistribution.hh"
#include "GateDistributionListManager.hh"

GateLocalEfficiencyMessenger::GateLocalEfficiencyMessenger(GateLocalEfficiency* itsPulseProcessor)
    : GatePulseProcessorMessenger(itsPulseProcessor)
{
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName() + "setEfficiency";
  crystalEfficiencyCmd = new G4UIcmdWithAString(cmdName,this);
  crystalEfficiencyCmd->SetGuidance("Set the efficiency");

  cmdName = GetDirectoryName() + "enableLevel";
  enableCommand = new G4UIcmdWithAnInteger(cmdName,this);
  enableCommand->SetGuidance("Set the efficiency");

  cmdName = GetDirectoryName() + "disableLevel";
  disableCommand = new G4UIcmdWithAnInteger(cmdName,this);
  disableCommand->SetGuidance("Set the efficiency");
}


GateLocalEfficiencyMessenger::~GateLocalEfficiencyMessenger()
{
  delete crystalEfficiencyCmd;
  delete enableCommand;
  delete disableCommand;
}


void GateLocalEfficiencyMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  GateLocalEfficiency* localEff = (GateLocalEfficiency*)GetNamedObject();
  if (!localEff) return;
  if ( command==crystalEfficiencyCmd ){
    GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
    if (distrib) localEff->SetEfficiency(distrib);}
  else if ( command==enableCommand )
    {localEff->SetMode(enableCommand->GetNewIntValue(newValue),true);}
  else if ( command==disableCommand)
    {localEff->SetMode(disableCommand->GetNewIntValue(newValue),false);}
  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
