/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateDistributionFileMessenger.hh"
#include "GateDistributionFile.hh"

#include "G4UIdirectory.hh"
#include <G4Tokenizer.hh>
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithoutParameter.hh"

// Constructor
GateDistributionFileMessenger::GateDistributionFileMessenger(GateDistributionFile* itsDistribution,
    			     const G4String& itsDirectoryName)
: GateDistributionArrayMessenger( itsDistribution,itsDirectoryName)
{
  G4String cmdName,guidance;

  cmdName = GetDirectoryName()+"setFileName";
  guidance = "Set the file name";
  setFileNameCmd = new G4UIcmdWithAString(cmdName,this);
  setFileNameCmd->SetGuidance(guidance);
  setFileNameCmd->SetParameterName("fileName",false);

  cmdName = GetDirectoryName()+"setColumnX";
  guidance = "Set the column describing the x values";
  setColXCmd = new G4UIcmdWithAnInteger(cmdName,this);
  setColXCmd->SetGuidance(guidance);
  setColXCmd->SetParameterName("colX",false);

  cmdName = GetDirectoryName()+"setColumnY";
  guidance = "Set the column describing the y values";
  setColYCmd = new G4UIcmdWithAnInteger(cmdName,this);
  setColYCmd->SetGuidance(guidance);
  setColYCmd->SetParameterName("colY",false);

  cmdName = GetDirectoryName()+"read";
  guidance = "Do read the file";
  readCmd = new G4UIcmdWithoutParameter(cmdName,this);
  readCmd->SetGuidance(guidance);
  cmdName = GetDirectoryName()+"readMatrix2d";
  guidance = "Do read the file";
  read2DCmd = new G4UIcmdWithoutParameter(cmdName,this);
  read2DCmd->SetGuidance(guidance);
  cmdName = GetDirectoryName()+"autoX";
  guidance = "Set automatic mode for X";
  autoXCmd = new G4UIcmdWithoutParameter(cmdName,this);
  autoXCmd->SetGuidance(guidance);
}



// Destructor
GateDistributionFileMessenger::~GateDistributionFileMessenger()
{
    delete readCmd;
    delete read2DCmd;
    delete autoXCmd;
    delete setColYCmd;
    delete setColXCmd;
    delete setFileNameCmd;
}



// UI command interpreter method
void GateDistributionFileMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if ( command==setFileNameCmd ){
    GetDistributionFile()->SetFileName(newValue);
  } else if( command==setColXCmd ) {
    G4int x = setColXCmd->GetNewIntValue(newValue);
    GetDistributionFile()->SetColumnX(x);
  } else if( command==autoXCmd ) {
    GetDistributionFile()->SetColumnX(-1);
    GetDistributionFile()->SetColumnY(0);
  } else if( command==setColYCmd ) {
    G4int y = setColXCmd->GetNewIntValue(newValue);
    GetDistributionFile()->SetColumnY(y);
  }   else if( command==readCmd ) {
    GetDistributionFile()->Read();
  }
 else if( command==read2DCmd ){
	  GetDistributionFile()->ReadMatrix2d();
  }
  else
    GateDistributionArrayMessenger::SetNewValue(command,newValue);
}
