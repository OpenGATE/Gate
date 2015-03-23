/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateReadoutMessenger.hh"

#include "GateReadout.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithAString.hh"

GateReadoutMessenger::GateReadoutMessenger(GateReadout* itsReadout)
    : GatePulseProcessorMessenger(itsReadout)
{
    G4String cmdName;

    // Command for choosing the depth of application of the readout
    cmdName = GetDirectoryName()+"setDepth";
    SetDepthCmd = new G4UIcmdWithAnInteger(cmdName,this);
    SetDepthCmd->SetGuidance("Defines the 'depth' of the readout:");
    SetDepthCmd->SetGuidance("pulses will be summed up if their volume IDs are identical up to this depth.");
    SetDepthCmd->SetGuidance("For instance, the default depth is 1: ");
    SetDepthCmd->SetGuidance("this means that pulses will be considered as taking place in a same block ");
    SetDepthCmd->SetGuidance("if their volume IDs are identical up to a depth of 1, i.e. the first two figures (depth 0 + depth 1)");
    // Command for choosing the policy to create the final pulse (S. Stute)
    // Note: in gate releases until v7.0 included, there was only one policy which was something like "TakeTheWinnerInEnergyForFinalPosition"
    //       we now introduce an option to choose a PMT like readout by using a policy like "TakeTheCentroidInEnergyForFinalPosition".
    //       So we add a command to choose between the two policy.
    cmdName = GetDirectoryName()+"setPolicy";
    SetPolicyCmd = new G4UIcmdWithAString(cmdName,this);
    SetPolicyCmd->SetGuidance("Defines the policy to be used to compute the final pulse position.");
    SetPolicyCmd->SetGuidance("There are three options: 'TakeWinner', 'TakeEnergyCentroid1' and 'TakeEnergyCentroid2'");
    SetPolicyCmd->SetGuidance("  --> 'TakeEnergyWinner': the final position will be the one for which the maximum energy was deposited");
    SetPolicyCmd->SetGuidance("  --> 'TakeEnergyCentroid': the energy centroid is computed based on crystal indices, meaning that the 'crystal' component of the system must be used. The depth is thus ignored.");
    SetPolicyCmd->SetGuidance("Note: when using the energyCentroid policy, the mother volume of the crystal level MUST NOT have any other daughter volumes declared BEFORE the crystal. Declaring it after is not a problem though.");
}


GateReadoutMessenger::~GateReadoutMessenger()
{
  delete SetDepthCmd;
  delete SetPolicyCmd;
}

void GateReadoutMessenger::SetNewValue(G4UIcommand* aCommand, G4String aString)
{
  if( aCommand==SetDepthCmd )
    { GetReadout()->SetDepth(SetDepthCmd->GetNewIntValue(aString)); }
  else if ( aCommand==SetPolicyCmd )
    { GetReadout()->SetPolicy(aString); }
  else
    GatePulseProcessorMessenger::SetNewValue(aCommand,aString);
}
