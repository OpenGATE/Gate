/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateDistributionExponentialMessenger.hh"
#include "GateDistributionExponential.hh"

#include "G4UIdirectory.hh"
#include <G4Tokenizer.hh>
#include "G4UIcmdWithADoubleAndUnit.hh"

// Constructor
GateDistributionExponentialMessenger::GateDistributionExponentialMessenger(GateDistributionExponential* itsDistribution,
    			     const G4String& itsDirectoryName)
: GateDistributionMessenger( itsDistribution,itsDirectoryName)
{
  G4String cmdName,guidance;

  cmdName = GetDirectoryName()+"setLambda";
  guidance = "Set the Exponential Lambda value";
  setLambdaCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  setLambdaCmd->SetGuidance(guidance);
  setLambdaCmd->SetParameterName("Lambda",false);

  cmdName = GetDirectoryName()+"setAmplitude";
  guidance = "Set the Exponential amplitude value";
  setAmplitudeCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  setAmplitudeCmd->SetGuidance(guidance);
  setAmplitudeCmd->SetParameterName("amplitude",false);

}



// Destructor
GateDistributionExponentialMessenger::~GateDistributionExponentialMessenger()
{
    delete setAmplitudeCmd;
    delete setLambdaCmd;
}



// UI command interpreter method
void GateDistributionExponentialMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if       ( command==setAmplitudeCmd ){
    G4double x = setAmplitudeCmd->GetNewDoubleValue(newValue);
    G4Tokenizer tok(newValue);
    G4String unit = tok();
    unit = tok();
    GetDistributionExponential()->SetAmplitude(x);
    SetUnitY(unit);
  } else if( command==setLambdaCmd ) {
    G4double x = setLambdaCmd->GetNewDoubleValue(newValue);
    G4Tokenizer tok(newValue);
    G4String unit = tok();
    unit = tok();
    GetDistributionExponential()->SetLambda(x);
    SetUnitX(unit);
  }
  else
    GateDistributionMessenger::SetNewValue(command,newValue);
}
