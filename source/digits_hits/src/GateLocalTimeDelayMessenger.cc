

#include "GateLocalTimeDelayMessenger.hh"
#include "GateLocalTimeDelay.hh"

#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"

GateLocalTimeDelayMessenger::GateLocalTimeDelayMessenger(GateLocalTimeDelay* itsDelay)
  : GatePulseProcessorMessenger(itsDelay)
{
  G4String guidance;
  G4String cmdName;
  m_count=0;

  cmdName = GetDirectoryName() + "chooseNewVolume";
  newVolCmd = new G4UIcmdWithAString(cmdName,this);
  newVolCmd->SetGuidance("Choose a volume for  TimeDelay");
}


GateLocalTimeDelayMessenger::~GateLocalTimeDelayMessenger()
{
  delete newVolCmd;
  for (G4int i=0;i<m_count;i++) {
    delete timeDelayCmd[i];
  }
}


void GateLocalTimeDelayMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
    if ( command==newVolCmd )
    {
        G4String cmdName2;

        if(GetLocalTimeDelay()->ChooseVolume(newValue) == 1) {
            m_volDirectory.push_back(new G4UIdirectory( (GetDirectoryName() + newValue + "/").c_str() ));
            m_volDirectory[m_count]->SetGuidance((G4String("characteristics of ") + newValue).c_str());

            m_name.push_back(newValue);


            cmdName2 = m_volDirectory[m_count]->GetCommandPath() + "setTimeDelay";
            timeDelayCmd.push_back(new G4UIcmdWithADoubleAndUnit(cmdName2,this));
            timeDelayCmd[m_count]->SetGuidance("Set the time delay for the chosen volume");
            timeDelayCmd[m_count]->SetUnitCategory("Time");

            m_count++;
        }
    }
    else
        SetNewValue2(command,newValue);
}

void GateLocalTimeDelayMessenger::SetNewValue2(G4UIcommand* command, G4String newValue)
{
  G4int test=0;
  for (G4int i=0;i<m_count;i++)  {
    if ( command==timeDelayCmd[i] ) {
      GetLocalTimeDelay()->SetDelay(m_name[i], timeDelayCmd[i]->GetNewDoubleValue(newValue));
      test=1;
    }
  }

  if(test==0)
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
