/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateReadoutMessenger.hh"

#include "GateReadout.hh"
#include "G4UIcmdWithAnInteger.hh"

GateReadoutMessenger::GateReadoutMessenger(GateReadout* itsReadout)
    : GatePulseProcessorMessenger(itsReadout)
{
    G4String cmdName;

    cmdName = GetDirectoryName()+"setDepth";
    SetDepthCmd = new G4UIcmdWithAnInteger(cmdName,this);
    SetDepthCmd->SetGuidance("Defines the 'depth' of the readout:");
    SetDepthCmd->SetGuidance("pulses will be summed up if their volume IDs are identical up to this depth.");
    SetDepthCmd->SetGuidance("For instance, the default depth is 1: ");
    SetDepthCmd->SetGuidance("this means that pulses will be considered as taking place in a same block ");
    SetDepthCmd->SetGuidance("if their volume IDs are identical up to a depth of 1, i.e. the first two figures (depth 0 + depth 1)");
}


GateReadoutMessenger::~GateReadoutMessenger()
{
  delete SetDepthCmd;
}

void GateReadoutMessenger::SetNewValue(G4UIcommand* aCommand, G4String aString)
{
  if( aCommand==SetDepthCmd )
    { GetReadout()->SetDepth(SetDepthCmd->GetNewIntValue(aString));}
  else
    GatePulseProcessorMessenger::SetNewValue(aCommand,aString);
}
