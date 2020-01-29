/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GatePulseAdderMessenger.hh"

#include "GatePulseAdder.hh"

GatePulseAdderMessenger::GatePulseAdderMessenger(GatePulseAdder* itsPulseAdder)
    : GatePulseProcessorMessenger(itsPulseAdder)
{

    G4String guidance;
    G4String cmdName;


    cmdName = GetDirectoryName()+"positionPolicy";
    positionPolicyCmd = new G4UIcmdWithAString(cmdName,this);
    positionPolicyCmd->SetGuidance("How to generate position");
    positionPolicyCmd->SetCandidates("energyWeightedCentroid takeEnergyWinner");

}

GatePulseAdderMessenger::~GatePulseAdderMessenger()
{
    delete   positionPolicyCmd;
}

void GatePulseAdderMessenger::SetNewValue(G4UIcommand* aCommand, G4String aString)
{
  if (aCommand ==positionPolicyCmd)
      { GetPulseAdder()->SetPositionPolicy(aString); }
    else
  GatePulseProcessorMessenger::SetNewValue(aCommand,aString);
}
