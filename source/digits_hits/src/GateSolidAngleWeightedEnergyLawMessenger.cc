/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  \class SolidAngleWeightEnergyLawMessenger
  This is a messenger for EnergyFraming digitizer module
  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/

#include "GateSolidAngleWeightedEnergyLawMessenger.hh"
#include "GateSolidAngleWeightedEnergyLaw.hh"


GateSolidAngleWeightedEnergyLawMessenger::GateSolidAngleWeightedEnergyLawMessenger(GateSolidAngleWeightedEnergyLaw* itsEffectiveEnergyLaw) :
    GateEffectiveEnergyLawMessenger(itsEffectiveEnergyLaw)
{


	G4String cmdName;
	G4String cmdName2;
    G4String cmdName3;

    cmdName = GetDirectoryName() + "setRentangleLengthX";
    szXCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    szXCmd ->SetGuidance("Set the length of the rectangle in X direction to calculate the solid angle subtended by the pulse");
    szXCmd->SetUnitCategory("Length");

    cmdName2 = GetDirectoryName() + "setRentangleLengthY";
    szYCmd = new G4UIcmdWithADoubleAndUnit(cmdName2,this);
    szYCmd ->SetGuidance("Set the length of the rectangle in Y direction to calculate the solid angle subtended by the pulse");
    szYCmd->SetUnitCategory("Length");


    cmdName3 = GetDirectoryName() + "setZSense4Readout";
    zSense4ReadoutCmd = new G4UIcmdWithAnInteger(cmdName3,this);
    zSense4ReadoutCmd->SetGuidance("Set Z sense for the ray entrance in Z reference frame (-1 or 1)");

}


GateSolidAngleWeightedEnergyLawMessenger::~GateSolidAngleWeightedEnergyLawMessenger() {
    delete szXCmd;
    delete szYCmd;
    delete zSense4ReadoutCmd;
}


GateSolidAngleWeightedEnergyLaw* GateSolidAngleWeightedEnergyLawMessenger::GetSolidAngleWeightedEnergyLaw() const {
    return dynamic_cast<GateSolidAngleWeightedEnergyLaw*>(GetEffectiveEnergyLaw());
}



void GateSolidAngleWeightedEnergyLawMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==szXCmd )
    { GetSolidAngleWeightedEnergyLaw()->SetRectangleSzX(szXCmd->GetNewDoubleValue(newValue)); }
  else if ( command==szYCmd )
    { GetSolidAngleWeightedEnergyLaw()->SetRectangleSzY(szYCmd->GetNewDoubleValue(newValue)); }
  else if ( command==zSense4ReadoutCmd )
    { GetSolidAngleWeightedEnergyLaw()->SetZSense(zSense4ReadoutCmd->GetNewIntValue(newValue));}

  else
    GateEffectiveEnergyLawMessenger::SetNewValue(command,newValue);
}
