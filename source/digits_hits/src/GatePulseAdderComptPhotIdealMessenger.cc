
#include "GatePulseAdderComptPhotIdealMessenger.hh"
#include "G4UIcmdWithABool.hh"
#include "GatePulseAdderComptPhotIdeal.hh"

GatePulseAdderComptPhotIdealMessenger::GatePulseAdderComptPhotIdealMessenger(GatePulseAdderComptPhotIdeal* itsPulseAdder)
    : GatePulseProcessorMessenger(itsPulseAdder)
{

       G4String cmdName;

       cmdName = GetDirectoryName() + "rejectEvtOtherProcesses";
       pRejectionPolicyCmd=new  G4UIcmdWithABool(cmdName,this);
       pRejectionPolicyCmd->SetGuidance("Set to 1 to reject events with at least one primary interaction different from C or P");

}


GatePulseAdderComptPhotIdealMessenger::~GatePulseAdderComptPhotIdealMessenger(){
    delete pRejectionPolicyCmd;
}

void GatePulseAdderComptPhotIdealMessenger::SetNewValue(G4UIcommand* aCommand, G4String aString)
{
  if ( aCommand==pRejectionPolicyCmd )
        { GetPulseAdderComptPhotIdeal()->SetEvtRejectionPolicy(pRejectionPolicyCmd->GetNewBoolValue(aString)); }
      else
  GatePulseProcessorMessenger::SetNewValue(aCommand,aString);
}
