

#include "GateLocalMultipleRejectionMessenger.hh"
#include "GateLocalMultipleRejection.hh"

#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"

GateLocalMultipleRejectionMessenger::GateLocalMultipleRejectionMessenger(GateLocalMultipleRejection* itsR)
  : GatePulseProcessorMessenger(itsR)
{
  G4String guidance;
  G4String cmdName;
  m_count=0;

  cmdName = GetDirectoryName() + "chooseNewVolume";
  newVolCmd = new G4UIcmdWithAString(cmdName,this);
  newVolCmd->SetGuidance("Choose a volume for  MultipleRejection");
}


GateLocalMultipleRejectionMessenger::~GateLocalMultipleRejectionMessenger()
{
  delete newVolCmd;
  for (G4int i=0;i<m_count;i++) {
     delete MultipleRejectionPolicyCmd[i];
     delete MultipleDefinitionCmd[i];
   }
}


void GateLocalMultipleRejectionMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
    if ( command==newVolCmd )
    {
        G4String cmdName2, cmdName3;

        if(GetLocalMultipleRejection()->ChooseVolume(newValue) == 1) {
            m_volDirectory.push_back(new G4UIdirectory( (GetDirectoryName() + newValue + "/").c_str() ));
            m_volDirectory[m_count]->SetGuidance((G4String("characteristics of ") + newValue).c_str());

            m_name.push_back(newValue);


            cmdName2 = m_volDirectory[m_count]->GetCommandPath() + "setEventRejection";
            MultipleRejectionPolicyCmd.push_back(new G4UIcmdWithABool(cmdName2,this));
            MultipleRejectionPolicyCmd[m_count]->SetGuidance("Set  rejection policy for the chosen volume. When there are multiples the whole event can be rejected or only those interactions in the studied volume ");



            cmdName3 = m_volDirectory[m_count]->GetCommandPath() + "setMultipleDefinition";
            MultipleDefinitionCmd.push_back(new G4UIcmdWithAString(cmdName3,this));
            MultipleDefinitionCmd[m_count]->SetGuidance("Set   the definition of multiples. We can considerer as multiples,  more than one single in the same volume Name or in the same volumeID (for repeaters). ");
            MultipleDefinitionCmd[m_count]->SetCandidates("volumeName volumeID");

            m_count++;
        }
    }
    else
        SetNewValue2(command,newValue);
}

void GateLocalMultipleRejectionMessenger::SetNewValue2(G4UIcommand* command, G4String newValue)
{
  G4int test=0;
  for (G4int i=0;i<m_count;i++)  {
    if ( command==MultipleRejectionPolicyCmd[i] ) {
       GetLocalMultipleRejection()->SetRejectionPolicy(m_name[i],  MultipleRejectionPolicyCmd[m_count]->GetNewBoolValue(newValue));
      test=1;
    }
  }
  if(test==0)
    for (G4int i=0;i<m_count;i++)  {
      if ( command==MultipleDefinitionCmd[i] ) {
         GetLocalMultipleRejection()->SetMultipleDefinition(m_name[i], newValue);
    test=1;

      }
    }

  if(test==0)
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
