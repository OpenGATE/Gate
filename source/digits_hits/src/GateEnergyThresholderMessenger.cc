/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateEnergyThresholderMessenger.hh"

#include "GateEnergyThresholder.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"

GateEnergyThresholderMessenger::GateEnergyThresholderMessenger(GateEnergyThresholder* itsEnergyThresholder)
    : GatePulseProcessorMessenger(itsEnergyThresholder)
{
  G4String guidance;
  G4String cmdName;
  G4String cmdName2;

  cmdName = GetDirectoryName() + "setThreshold";
  thresholdCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  thresholdCmd->SetGuidance("Set threshold (in keV) for pulse-discrimination");
  thresholdCmd->SetUnitCategory("Energy");


  cmdName2 = GetDirectoryName() + "setLaw";
  lawCmd = new G4UIcmdWithAString(cmdName2,this);
  lawCmd->SetGuidance("Set the law of effective energy  for the threshold");
}


GateEnergyThresholderMessenger::~GateEnergyThresholderMessenger()
{
  delete thresholdCmd;
   delete lawCmd;
}


GateVEffectiveEnergyLaw* GateEnergyThresholderMessenger::CreateEffectiveEnergyLaw(const G4String& law) {

    if ( law == "solidAngleWeighted" ) {
        return new GateSolidAngleWeightedEnergyLaw(GetEnergyThresholder()->GetObjectName()+ G4String("/solidAngleWeighted"));

    } else if ( law == "depositedEnergy" ) {
        return new GateDepositedEnergyLaw(GetEnergyThresholder()->GetObjectName() + G4String("/depositedEnergy"));
    } else {
        G4cerr << "No match for '" << law << "' effective energy law.\n";
        G4cerr << "Candidates are: solidAngleWeighted  depositedEnergy\n";
    }

    return NULL;
}

void GateEnergyThresholderMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==thresholdCmd )
    { GetEnergyThresholder()->SetThreshold(thresholdCmd->GetNewDoubleValue(newValue)); }
  else if (command==lawCmd ){
      GateVEffectiveEnergyLaw* a_energyLaw = CreateEffectiveEnergyLaw(newValue);
              if (a_energyLaw != NULL) {
                  GetEnergyThresholder()->SetEffectiveEnergyLaw(a_energyLaw);
              }
  }

  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
