/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateLightYieldMessenger.hh"
#include "GateLightYield.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADouble.hh"

GateLightYieldMessenger::GateLightYieldMessenger(GateLightYield* itsLightYield)
  : GatePulseProcessorMessenger(itsLightYield)
{
  G4String guidance;
  G4String cmdName;
  m_count=0;

  cmdName = GetDirectoryName() + "chooseNewVolume";
  newVolCmd = new G4UIcmdWithAString(cmdName,this);
  newVolCmd->SetGuidance("Choose a volume for light yield");
}


GateLightYieldMessenger::~GateLightYieldMessenger()
{
  delete newVolCmd;
  for (G4int i=0;i<m_count;i++) {
    delete lightOutputCmd[i];
  }
}


void GateLightYieldMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==newVolCmd )
    {
      G4String cmdName2;

      if(GetLightYield()->ChooseVolume(newValue) == 1) {
	m_volDirectory.push_back(new G4UIdirectory( (GetDirectoryName() + newValue + "/").c_str() ));
	m_volDirectory[m_count]->SetGuidance((G4String("light yield of ") + newValue).c_str());

	m_name.push_back(newValue);

	cmdName2 = m_volDirectory[m_count]->GetCommandPath() + "setLightOutput";
	lightOutputCmd.push_back(new G4UIcmdWithADouble(cmdName2,this));
	lightOutputCmd[m_count]->SetGuidance("Set the Light Output for this crystal (ph/MeV):");

	m_count++;
      }
    }
  else
    SetNewValue2(command,newValue);
}

void GateLightYieldMessenger::SetNewValue2(G4UIcommand* command, G4String newValue)
{
  G4int test=0;
  for (G4int i=0;i<m_count;i++)  {
    if ( command==lightOutputCmd[i] ) {
      GetLightYield()->SetLightOutput(m_name[i], lightOutputCmd[i]->GetNewDoubleValue(newValue)/MeV);
      test=1;
    }
  }
  if(test==0)
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
