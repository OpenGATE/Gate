/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCrystalBlurringMessenger.hh"

#include "GateCrystalBlurring.hh"

#include  "G4UIcmdWithADoubleAndUnit.hh"

#include "G4UIcmdWithADouble.hh"

GateCrystalBlurringMessenger::GateCrystalBlurringMessenger(GateCrystalBlurring* itsCrystalresolution)
    : GatePulseProcessorMessenger(itsCrystalresolution)
{
  G4String guidance;
  G4String cmdName;
  G4String cmdName1;
  G4String cmdName2;
  G4String cmdName3;

  cmdName = GetDirectoryName() + "setCrystalResolutionMin";
  crystalresolutionminCmd = new G4UIcmdWithADouble(cmdName,this);
  crystalresolutionminCmd->SetGuidance("Set the minimum resolution in energie, crystal by crystal, for gaussian blurring");

  cmdName1 = GetDirectoryName() + "setCrystalResolutionMax";
  crystalresolutionmaxCmd = new G4UIcmdWithADouble(cmdName1,this);
  crystalresolutionmaxCmd->SetGuidance("Set the maximum resolution in energie, crystal by crystal, for gaussian blurring");

  cmdName2 = GetDirectoryName() + "setCrystalQE";
  crystalQECmd = new G4UIcmdWithADouble(cmdName2,this);
  crystalQECmd->SetGuidance("Set the variation of quantum efficency crystal by crystal");

  cmdName3 = GetDirectoryName() + "setCrystalEnergyOfReference";
  crystalerefCmd = new G4UIcmdWithADoubleAndUnit(cmdName3,this);
  crystalerefCmd->SetGuidance("Set the energy of reference (in keV) for the selected resolution");
  crystalerefCmd->SetUnitCategory("Energy");
}


GateCrystalBlurringMessenger::~GateCrystalBlurringMessenger()
{
  delete crystalresolutionminCmd;
  delete crystalresolutionmaxCmd;
  delete crystalQECmd;
  delete crystalerefCmd;
}


void GateCrystalBlurringMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==crystalresolutionminCmd )
    { GetCrystalBlurring()->SetCrystalResolutionMin(crystalresolutionminCmd->GetNewDoubleValue(newValue)); }
  else if ( command==crystalresolutionmaxCmd)
    { GetCrystalBlurring()->SetCrystalResolutionMax(crystalresolutionmaxCmd->GetNewDoubleValue(newValue)); }
  else if ( command==crystalQECmd)
    { GetCrystalBlurring()->SetCrystalQE(crystalQECmd->GetNewDoubleValue(newValue)); }
  else if ( command==crystalerefCmd )
    { GetCrystalBlurring()->SetCrystalRefEnergy(crystalerefCmd->GetNewDoubleValue(newValue)); }
  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
