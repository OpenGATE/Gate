


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
