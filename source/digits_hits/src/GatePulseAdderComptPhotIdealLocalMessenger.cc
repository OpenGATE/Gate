
#include "GatePulseAdderComptPhotIdealLocalMessenger.hh"

#include "GatePulseAdderComptPhotIdealLocal.hh"

GatePulseAdderComptPhotIdealLocalMessenger::GatePulseAdderComptPhotIdealLocalMessenger(GatePulseAdderComptPhotIdealLocal* itsPulseAdder)
    : GatePulseProcessorMessenger(itsPulseAdder)
{

    G4String cmdName;

    cmdName = GetDirectoryName() + "chooseNewVolume";
    newVolCmd = new G4UIcmdWithAString(cmdName,this);
    newVolCmd->SetGuidance("Choose a volume for  idealAdder");
}

GatePulseAdderComptPhotIdealLocalMessenger::~GatePulseAdderComptPhotIdealLocalMessenger(){
     delete newVolCmd;
}

void GatePulseAdderComptPhotIdealLocalMessenger::SetNewValue(G4UIcommand* aCommand, G4String aString)
{

  if ( aCommand==newVolCmd )
  {
       GetPulseAdderComptPhotIdealLocal()->ChooseVolume(aString);

  }
  else{
     GatePulseProcessorMessenger::SetNewValue(aCommand,aString);

  }
}


