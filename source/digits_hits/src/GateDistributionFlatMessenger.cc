/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateDistributionFlatMessenger.hh"
#include "GateDistributionFlat.hh"

#include "G4UIdirectory.hh"
#include <G4Tokenizer.hh>
#include "G4UIcmdWithADoubleAndUnit.hh"

// Constructor
GateDistributionFlatMessenger::GateDistributionFlatMessenger(GateDistributionFlat* itsDistribution,
    			     const G4String& itsDirectoryName)
: GateDistributionMessenger( itsDistribution,itsDirectoryName)
{
  G4String cmdName,guidance;

  cmdName = GetDirectoryName()+"setMin";
  guidance = "Set the Flat min value";
  setMinCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  setMinCmd->SetGuidance(guidance);
  setMinCmd->SetParameterName("min",false);

  cmdName = GetDirectoryName()+"setMax";
  guidance = "Set the Flat RMS value";
  setMaxCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  setMaxCmd->SetGuidance(guidance);
  setMaxCmd->SetParameterName("Max",false);

  cmdName = GetDirectoryName()+"setAmplitude";
  guidance = "Set the Flat amplitude value";
  setAmplitudeCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  setAmplitudeCmd->SetGuidance(guidance);
  setAmplitudeCmd->SetParameterName("amplitude",false);

}



// Destructor
GateDistributionFlatMessenger::~GateDistributionFlatMessenger()
{
    delete setAmplitudeCmd;
    delete setMaxCmd;
    delete setMinCmd;
}



// UI command interpreter method
void GateDistributionFlatMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if       ( command==setAmplitudeCmd ){
    G4double x = setAmplitudeCmd->GetNewDoubleValue(newValue);
    G4Tokenizer tok(newValue);
    G4String unit = tok();
    unit = tok();
    GetDistributionFlat()->SetAmplitude(x);
    SetUnitY(unit);
  } else if( command==setMinCmd ) {
    G4double x = setMinCmd->GetNewDoubleValue(newValue);
    G4Tokenizer tok(newValue);
    G4String unit = tok();
    unit = tok();
    GetDistributionFlat()->SetMin(x);
    SetUnitX(unit);
  } else if( command==setMaxCmd ) {
    G4double x = setMaxCmd->GetNewDoubleValue(newValue);
    G4Tokenizer tok(newValue);
    G4String unit = tok();
    unit = tok();
    GetDistributionFlat()->SetMax(x);
    SetUnitX(unit);
  }
  else
    GateDistributionMessenger::SetNewValue(command,newValue);
}
