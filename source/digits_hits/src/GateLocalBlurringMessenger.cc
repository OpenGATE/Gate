/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateLocalBlurringMessenger.hh"
#include "GateLocalBlurring.hh"

#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"

GateLocalBlurringMessenger::GateLocalBlurringMessenger(GateLocalBlurring* itsResolution)
  : GatePulseProcessorMessenger(itsResolution)
{
  G4String guidance;
  G4String cmdName;
  m_count=0;

  cmdName = GetDirectoryName() + "chooseNewVolume";
  newVolCmd = new G4UIcmdWithAString(cmdName,this);
  newVolCmd->SetGuidance("Choose a volume for gaussian blurring");
}


GateLocalBlurringMessenger::~GateLocalBlurringMessenger()
{
  delete newVolCmd;
  for (G4int i=0;i<m_count;i++) {
    delete resolutionCmd[i];
    delete erefCmd[i];
  }
}


void GateLocalBlurringMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==newVolCmd )
    {
      G4String cmdName2, cmdName3;

      if(GetLocalBlurring()->ChooseVolume(newValue) == 1) {
	m_volDirectory.push_back(new G4UIdirectory( (GetDirectoryName() + newValue + "/").c_str() ));
	m_volDirectory[m_count]->SetGuidance((G4String("characteristics of ") + newValue).c_str());

	m_name.push_back(newValue);

	cmdName2 = m_volDirectory[m_count]->GetCommandPath() + "setResolution";
	resolutionCmd.push_back(new G4UIcmdWithADouble(cmdName2,this));
	resolutionCmd[m_count]->SetGuidance("Set the resolution in energie for gaussian blurring");

	cmdName3 = m_volDirectory[m_count]->GetCommandPath() + "setEnergyOfReference";
	erefCmd.push_back(new G4UIcmdWithADoubleAndUnit(cmdName3,this));
	erefCmd[m_count]->SetGuidance("Set the energy of reference (in keV) for the selected resolution");
	erefCmd[m_count]->SetUnitCategory("Energy");

	m_count++;
      }
    }
  else
    SetNewValue2(command,newValue);
}

void GateLocalBlurringMessenger::SetNewValue2(G4UIcommand* command, G4String newValue)
{
  G4int test=0;
  for (G4int i=0;i<m_count;i++)  {
    if ( command==resolutionCmd[i] ) {
      GetLocalBlurring()->SetResolution(m_name[i], resolutionCmd[i]->GetNewDoubleValue(newValue));
      test=1;
    }
  }
  if(test==0)
    for (G4int i=0;i<m_count;i++)  {
      if ( command==erefCmd[i] ) {
	GetLocalBlurring()->SetRefEnergy(m_name[i], erefCmd[i]->GetNewDoubleValue(newValue));
	test=1;
      }
    }
  if(test==0)
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
