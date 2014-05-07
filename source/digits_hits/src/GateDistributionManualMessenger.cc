/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateDistributionManualMessenger.hh"
#include "GateDistributionManual.hh"

#include "G4UIdirectory.hh"
#include <G4Tokenizer.hh>
#include "GateUIcmdWithTwoDouble.hh"
#include "G4UIcmdWithADouble.hh"

// Constructor
GateDistributionManualMessenger::GateDistributionManualMessenger(GateDistributionManual* itsDistribution,
    			     const G4String& itsDirectoryName)
: GateDistributionArrayMessenger( itsDistribution,itsDirectoryName)
{
  G4String cmdName,guidance;

  cmdName = GetDirectoryName()+"insertPoint";
  guidance = "Insert a point";
  insPointCmd = new GateUIcmdWithTwoDouble(cmdName,this);
  insPointCmd->SetGuidance(guidance);

  cmdName = GetDirectoryName()+"addPoint";
  guidance = "Add a point (automatic X)";
  addPointCmd = new G4UIcmdWithADouble(cmdName,this);
  addPointCmd->SetGuidance(guidance);
}



// Destructor
GateDistributionManualMessenger::~GateDistributionManualMessenger()
{
    delete addPointCmd;
    delete insPointCmd;
}



// UI command interpreter method
void GateDistributionManualMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if ( command==insPointCmd ){
    G4double x = insPointCmd->GetNewDoubleValue(0,newValue);
    G4double y = insPointCmd->GetNewDoubleValue(1,newValue);
    GetDistributionManual()->AddPoint(x,y);
  } else if ( command==addPointCmd ){
    G4double y = addPointCmd->GetNewDoubleValue(newValue);
    GetDistributionManual()->AddPoint(y);
  }
  else
    GateDistributionArrayMessenger::SetNewValue(command,newValue);
}
