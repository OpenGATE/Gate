/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateTransferEfficiencyMessenger.hh"
#include "GateTransferEfficiency.hh"

#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADouble.hh"

GateTransferEfficiencyMessenger::GateTransferEfficiencyMessenger(GateTransferEfficiency* itsTE)
  : GatePulseProcessorMessenger(itsTE)
{
  G4String guidance;
  G4String cmdName;
  m_count=0;

  cmdName = GetDirectoryName() + "chooseNewVolume";
  newVolCmd = new G4UIcmdWithAString(cmdName,this);
  newVolCmd->SetGuidance("Choose a volume for transfer efficiency");
}


GateTransferEfficiencyMessenger::~GateTransferEfficiencyMessenger()
{
  delete newVolCmd;
  for (G4int i=0;i<m_count;i++) {
    delete coeffTECmd[i];
  }
}


void GateTransferEfficiencyMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==newVolCmd )
    {
      G4String cmdName2;

      if(GetTransferEfficiency()->ChooseVolume(newValue) == 1) {
	m_volDirectory.push_back(new G4UIdirectory( (GetDirectoryName() + newValue + "/").c_str() ));
	m_volDirectory[m_count]->SetGuidance((G4String("Transfert efficiency for") + newValue).c_str());

	m_name.push_back(newValue);

	cmdName2 = m_volDirectory[m_count]->GetCommandPath() + "setTECoef";
	coeffTECmd.push_back(new G4UIcmdWithADouble(cmdName2,this));
	coeffTECmd[m_count]->SetGuidance("Set the coefficient for transfer efficiency");

	m_count++;
      }
    }
  else
    SetNewValue2(command,newValue);
}

void GateTransferEfficiencyMessenger::SetNewValue2(G4UIcommand* command, G4String newValue)
{
  G4int test=0;
  for (G4int i=0;i<m_count;i++)  {
    if ( command==coeffTECmd[i] ) {
      GetTransferEfficiency()->SetTECoeff(m_name[i], coeffTECmd[i]->GetNewDoubleValue(newValue));
      test=1;
    }
  }
  if(test==0)
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
