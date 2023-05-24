/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*

  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com

*/


#include "GateDualLayerLawMessenger.hh"
#include "GateDualLayerLaw.hh"


GateDualLayerLawMessenger::GateDualLayerLawMessenger(GateDualLayerLaw* itsDualLayerLaw) :
    GateDoILawMessenger(itsDualLayerLaw)
{


}





GateDualLayerLaw* GateDualLayerLawMessenger::GetDualLayerLaw() const {
    return dynamic_cast<GateDualLayerLaw*>(GetDoILaw());
}









void GateDualLayerLawMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{


    GateDoILawMessenger::SetNewValue(command,newValue);
}
