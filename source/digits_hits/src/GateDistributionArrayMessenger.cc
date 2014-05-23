/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateDistributionArrayMessenger.hh"
#include "GateVDistributionArray.hh"

#include "G4UIdirectory.hh"
#include <G4Tokenizer.hh>
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"

// Constructor
GateDistributionArrayMessenger::GateDistributionArrayMessenger(GateVDistributionArray* itsDistribution,
    			     const G4String& itsDirectoryName)
: GateDistributionMessenger( itsDistribution,itsDirectoryName)
{
  G4String cmdName,guidance;

  cmdName = GetDirectoryName()+"setUnitX";
  guidance = "Specify the unit of X axis";
  setUnitX_Cmd = new G4UIcmdWithAString(cmdName,this);
  setUnitX_Cmd->SetGuidance(guidance);
  setUnitX_Cmd->SetParameterName("unit",false);

  cmdName = GetDirectoryName()+"setUnitY";
  guidance = "Specify the unit of Y axis";
  setUnitY_Cmd = new G4UIcmdWithAString(cmdName,this);
  setUnitY_Cmd->SetGuidance(guidance);
  setUnitY_Cmd->SetParameterName("unit",false);

  cmdName = GetDirectoryName()+"autoXstart";
  guidance = "Specify the first int used, if automatic X mode";
  setAutoX_Cmd = new G4UIcmdWithAnInteger(cmdName,this);
  setAutoX_Cmd->SetGuidance(guidance);
  setAutoX_Cmd->SetParameterName("start",false);


}



// Destructor
GateDistributionArrayMessenger::~GateDistributionArrayMessenger()
{
    delete setUnitY_Cmd;
    delete setUnitX_Cmd;
    delete setAutoX_Cmd;
}



// UI command interpreter method
void GateDistributionArrayMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if( command==setUnitX_Cmd ) {
    SetUnitX(newValue);
    G4double factorX = G4UIcommand::ValueOf(newValue);
    if (factorX==0) factorX=1;
    GetVDistributionArray()->SetFactorX(1./factorX);
  } else if( command==setUnitY_Cmd ) {
    SetUnitY(newValue);
    G4double factorY = G4UIcommand::ValueOf(newValue);
    if (factorY==0) factorY=1;
    GetVDistributionArray()->SetFactorY(1./factorY);
  } else if( command==setAutoX_Cmd ) {
    G4int start = setAutoX_Cmd->GetNewIntValue(newValue);
    GetVDistributionArray()->SetAutoStart(start);
  }
  else
    GateDistributionMessenger::SetNewValue(command,newValue);
}
