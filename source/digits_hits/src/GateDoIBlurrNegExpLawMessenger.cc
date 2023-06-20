/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  \class  GateDoIBlurrNegExpLawMessenger

  The user can choose the value of ExpInvDecayConst and FWHM needed for the crystal entrance.

  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#include "GateDoIBlurrNegExpLawMessenger.hh"
#include "GateDoIBlurrNegExpLaw.hh"


GateDoIBlurrNegExpLawMessenger::GateDoIBlurrNegExpLawMessenger(GateDoIBlurrNegExpLaw* itsDoIBlurrNegExpLaw) :
    GateDoILawMessenger(itsDoIBlurrNegExpLaw)
{

    G4String cmdName;
    G4String cmdName2;

    cmdName = GetDirectoryName() + "setExpInvDecayConst";
    expDCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    expDCmd ->SetGuidance("Set the exponential  inv decay constant in Length unit");
    expDCmd->SetUnitCategory("Length");


    cmdName2 = GetDirectoryName() + "setCrysEntranceFWHM";
    entFWHMCmd = new G4UIcmdWithADoubleAndUnit(cmdName2,this);
    entFWHMCmd->SetGuidance("Set the value of the FWHM at crystal entrance (far from the photodetector)");
    entFWHMCmd->SetUnitCategory("Length");

}



  GateDoIBlurrNegExpLawMessenger::~GateDoIBlurrNegExpLawMessenger(){
    delete  entFWHMCmd;
    delete  expDCmd;
}

GateDoIBlurrNegExpLaw* GateDoIBlurrNegExpLawMessenger::GetDoIBlurrNegExpLaw() const {
    return dynamic_cast<GateDoIBlurrNegExpLaw*>(GetDoILaw());
}





void GateDoIBlurrNegExpLawMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{

    if ( command==expDCmd )
      { GetDoIBlurrNegExpLaw()->SetExpInvDecayConst(expDCmd->GetNewDoubleValue(newValue)); }
    else if ( command==entFWHMCmd )
      { GetDoIBlurrNegExpLaw()->SetEntranceFWHM(entFWHMCmd->GetNewDoubleValue(newValue)); }

    else

         GateDoILawMessenger::SetNewValue(command,newValue);
}
