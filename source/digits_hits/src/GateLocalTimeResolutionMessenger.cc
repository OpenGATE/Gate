

#include "GateLocalTimeResolutionMessenger.hh"
#include "GateLocalTimeResolution.hh"

#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"

GateLocalTimeResolutionMessenger::GateLocalTimeResolutionMessenger(GateLocalTimeResolution* itsResol)
  : GatePulseProcessorMessenger(itsResol)
{
  G4String guidance;
  G4String cmdName;
  m_count=0;

  cmdName = GetDirectoryName() + "chooseNewVolume";
  newVolCmd = new G4UIcmdWithAString(cmdName,this);
  newVolCmd->SetGuidance("Choose a volume for  TimeResolution");
}


GateLocalTimeResolutionMessenger::~GateLocalTimeResolutionMessenger()
{
  delete newVolCmd;
  for (G4int i=0;i<m_count;i++) {
    delete timeResolutionCmd[i];
  }
}


void GateLocalTimeResolutionMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
    if ( command==newVolCmd )
    {
        G4String cmdName2;

        if(GetLocalTimeResolution()->ChooseVolume(newValue) == 1) {
            m_volDirectory.push_back(new G4UIdirectory( (GetDirectoryName() + newValue + "/").c_str() ));
            m_volDirectory[m_count]->SetGuidance((G4String("characteristics of ") + newValue).c_str());

            m_name.push_back(newValue);


            cmdName2 = m_volDirectory[m_count]->GetCommandPath() + "setTimeResolution";
            timeResolutionCmd.push_back(new G4UIcmdWithADoubleAndUnit(cmdName2,this));
            timeResolutionCmd[m_count]->SetGuidance("Set the time resolution (FWHM) for the chosen volume");
            timeResolutionCmd[m_count]->SetUnitCategory("Time");

            m_count++;
        }
    }
    else
        SetNewValue2(command,newValue);
}

void GateLocalTimeResolutionMessenger::SetNewValue2(G4UIcommand* command, G4String newValue)
{
  G4int test=0;
  for (G4int i=0;i<m_count;i++)  {
    if ( command==timeResolutionCmd[i] ) {
      GetLocalTimeResolution()->SetTimeResolution(m_name[i], timeResolutionCmd[i]->GetNewDoubleValue(newValue));
      test=1;
    }
  }

  if(test==0)
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
