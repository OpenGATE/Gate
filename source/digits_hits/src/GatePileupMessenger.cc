/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GatePileupMessenger.hh"

#include "GatePileup.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

GatePileupMessenger::GatePileupMessenger(GatePileup* itsPileup)
    : GatePulseProcessorMessenger(itsPileup)
{
    G4String cmdName;

    cmdName = GetDirectoryName()+"setDepth";
    SetDepthCmd = new G4UIcmdWithAnInteger(cmdName,this);
    SetDepthCmd->SetGuidance("Defines the 'depth' of the Pileup");

    cmdName = GetDirectoryName()+"setPileup";
    SetPileupCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    SetPileupCmd->SetGuidance("Defines the 'time' of the Pileup");
    SetPileupCmd->SetUnitCategory("Time");

}


GatePileupMessenger::~GatePileupMessenger()
{
  delete SetDepthCmd;
  delete SetPileupCmd;
}

void GatePileupMessenger::SetNewValue(G4UIcommand* aCommand, G4String aString)
{
  if( aCommand==SetDepthCmd )
    { GetPileup()->SetDepth(SetDepthCmd->GetNewIntValue(aString));}
  else if (aCommand==SetPileupCmd )
    { GetPileup()->SetPileup(SetPileupCmd->GetNewDoubleValue(aString));}
  else
    GatePulseProcessorMessenger::SetNewValue(aCommand,aString);
}
