/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateBlurringWithIntrinsicResolutionMessenger.hh"
#include "GateBlurringWithIntrinsicResolution.hh"

#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"

GateBlurringWithIntrinsicResolutionMessenger::GateBlurringWithIntrinsicResolutionMessenger(GateBlurringWithIntrinsicResolution* itsIntrinsic)
  : GatePulseProcessorMessenger(itsIntrinsic)
{
  G4String guidance;
  G4String cmdName;
  m_count=0;

  cmdName = GetDirectoryName() + "chooseNewVolume";
  newVolCmd = new G4UIcmdWithAString(cmdName,this);
  newVolCmd->SetGuidance("Choose a volume for intrinsic blurring");
}


GateBlurringWithIntrinsicResolutionMessenger::~GateBlurringWithIntrinsicResolutionMessenger()
{
  delete newVolCmd;
  for (G4int i=0;i<m_count;i++) {
    delete resolutionCmd[i];
    delete erefCmd[i];
  }
}


void GateBlurringWithIntrinsicResolutionMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==newVolCmd )
    {
      G4String cmdName2, cmdName3;

      if(GetBlurringWithIntrinsicResolution()->ChooseVolume(newValue) == 1) {
	m_volDirectory.push_back(new G4UIdirectory( (GetDirectoryName() + newValue + "/").c_str() ));
	m_volDirectory[m_count]->SetGuidance((G4String("characteristics of ") + newValue).c_str());

	m_name.push_back(newValue);

	cmdName2 = m_volDirectory[m_count]->GetCommandPath() + "setIntrinsicResolution";
	resolutionCmd.push_back(new G4UIcmdWithADouble(cmdName2,this));
	resolutionCmd[m_count]->SetGuidance("Set the intrinsic resolution in energie for this crystal");

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

void GateBlurringWithIntrinsicResolutionMessenger::SetNewValue2(G4UIcommand* command, G4String newValue)
{
  G4int test=0;
  for (G4int i=0;i<m_count;i++)  {
    if ( command==resolutionCmd[i] ) {
      GetBlurringWithIntrinsicResolution()->SetIntrinsicResolution(m_name[i], resolutionCmd[i]->GetNewDoubleValue(newValue));
      test=1;
    }
  }
  if(test==0)
    for (G4int i=0;i<m_count;i++)  {
      if ( command==erefCmd[i] ) {
	GetBlurringWithIntrinsicResolution()->SetRefEnergy(m_name[i], erefCmd[i]->GetNewDoubleValue(newValue));
	test=1;
      }
    }
  if(test==0)
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
