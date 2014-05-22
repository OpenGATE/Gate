/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateRunManagerMessenger.hh"
#include "GateRunManager.hh"

#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"
#include "GateDetectorConstruction.hh"

//----------------------------------------------------------------------------------------
GateRunManagerMessenger::GateRunManagerMessenger(GateRunManager* runManager)
  : pRunManager(runManager)
{
  pRunInitCmd = new G4UIcmdWithoutParameter("/gate/run/initialize",this);
  pRunInitCmd->SetGuidance("Initialize geometry, actors and physics list.");

  pRunGeomUpdateCmd = new G4UIcmdWithoutParameter("/gate/geometry/rebuild",this);
  pRunGeomUpdateCmd->SetGuidance("Rebuild the whole geometry.");

  pRunEnableGlobalOutputCmd = new G4UIcmdWithABool("/gate/run/enableGlobalOutput",this);
  pRunEnableGlobalOutputCmd->SetGuidance("Enabled by default. Use 'false' only for applications that do not use 'systems' (PET, SPECT etc), it will be a bit faster.");
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
GateRunManagerMessenger::~GateRunManagerMessenger()
{
  delete pRunInitCmd;
  delete pRunGeomUpdateCmd;
  delete pRunEnableGlobalOutputCmd;
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
void GateRunManagerMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{   
  if (command == pRunInitCmd) {   
    pRunManager->InitializeAll();
  }
  else if (command == pRunGeomUpdateCmd) {
    pRunManager->InitGeometryOnly();
  }
  else if (command == pRunEnableGlobalOutputCmd) {
    pRunManager->EnableGlobalOutput(pRunEnableGlobalOutputCmd->GetNewBoolValue(newValue));
  }
}
//----------------------------------------------------------------------------------------
