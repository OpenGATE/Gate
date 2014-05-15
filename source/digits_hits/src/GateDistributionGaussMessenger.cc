/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateDistributionGaussMessenger.hh"
#include "GateDistributionGauss.hh"

#include "G4UIdirectory.hh"
#include <G4Tokenizer.hh>
#include "G4UIcmdWithADoubleAndUnit.hh"

// Constructor
GateDistributionGaussMessenger::GateDistributionGaussMessenger(GateDistributionGauss* itsDistribution,
    			     const G4String& itsDirectoryName)
: GateDistributionMessenger( itsDistribution,itsDirectoryName)
{
  G4String cmdName,guidance;

  cmdName = GetDirectoryName()+"setMean";
  guidance = "Set the gauss mean value";
  setMeanCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  setMeanCmd->SetGuidance(guidance);
  setMeanCmd->SetParameterName("mean",false);

  cmdName = GetDirectoryName()+"setSigma";
  guidance = "Set the gauss RMS value";
  setSigmaCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  setSigmaCmd->SetGuidance(guidance);
  setSigmaCmd->SetParameterName("sigma",false);

  cmdName = GetDirectoryName()+"setAmplitude";
  guidance = "Set the gauss amplitude value";
  setAmplitudeCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  setAmplitudeCmd->SetGuidance(guidance);
  setAmplitudeCmd->SetParameterName("amplitude",false);

}



// Destructor
GateDistributionGaussMessenger::~GateDistributionGaussMessenger()
{
    delete setAmplitudeCmd;
    delete setSigmaCmd;
    delete setMeanCmd;
}



// UI command interpreter method
void GateDistributionGaussMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if       ( command==setAmplitudeCmd ){
    G4double x = setAmplitudeCmd->GetNewDoubleValue(newValue);
    G4Tokenizer tok(newValue);
    G4String unit = tok();
    unit = tok();
    GetDistributionGauss()->SetAmplitude(x);
    SetUnitY(unit);
  } else if( command==setMeanCmd ) {
    G4double x = setMeanCmd->GetNewDoubleValue(newValue);
    G4Tokenizer tok(newValue);
    G4String unit = tok();
    unit = tok();
    GetDistributionGauss()->SetMean(x);
    SetUnitX(unit);
  } else if( command==setSigmaCmd ) {
    G4double x = setSigmaCmd->GetNewDoubleValue(newValue);
    G4Tokenizer tok(newValue);
    G4String unit = tok();
    unit = tok();
    GetDistributionGauss()->SetSigma(x);
    SetUnitX(unit);
  }
  else
    GateDistributionMessenger::SetNewValue(command,newValue);
}
