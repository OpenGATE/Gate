/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateCCCoincidenceSequenceReconMessenger.hh"

#include "GateCCCoincidenceSequenceRecon.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"

GateCCCoincidenceSequenceReconMessenger::GateCCCoincidenceSequenceReconMessenger(GateCCCoincidenceSequenceRecon* itsSequence)
    : GateClockDependentMessenger(itsSequence)
{
    G4String guidance;
    G4String cmdName;


    cmdName = GetDirectoryName()+"setSequencePolicy";
    sequencePolicyCmd = new G4UIcmdWithAString(cmdName,this);
    sequencePolicyCmd->SetGuidance("How to order coincidences");
    sequencePolicyCmd->SetCandidates("singlesTime lowestEnergyFirst randomly axialDist2Source");
   // sequencePolicyCmd->SetCandidates("singlesTime lowestEnergyFirst randomly axialDist2Source revanC_CSR");



}


GateCCCoincidenceSequenceReconMessenger::~GateCCCoincidenceSequenceReconMessenger()
{

    delete sequencePolicyCmd;

}


void GateCCCoincidenceSequenceReconMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
    if (command == sequencePolicyCmd)
    { GetCoincidenceSequence()->SetSequencePolicy(newValue); }
    else
        GateClockDependentMessenger::SetNewValue(command,newValue);
}
