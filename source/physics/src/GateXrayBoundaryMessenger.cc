/*##########################################
#developed by Zhenjie Cen
#
#CREATIS
#
#May 2016
##########################################
*/
#include "GateXrayBoundaryMessenger.hh"
#include "GateVProcess.hh"

GateXrayBoundaryMessenger::GateXrayBoundaryMessenger(GateVProcess *pb):GateVProcessMessenger(pb)
{
    BuildCommands("processes/" + pb->GetG4ProcessName());
}

GateXrayBoundaryMessenger::~GateXrayBoundaryMessenger() {}

void GateXrayBoundaryMessenger::BuildCommands(G4String base) {G4cout << base;}

void GateXrayBoundaryMessenger::SetNewValue(G4UIcommand* , G4String ) {}

