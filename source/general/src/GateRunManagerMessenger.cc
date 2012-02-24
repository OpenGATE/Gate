/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateRunManagerMessenger.hh"
#include "GateRunManager.hh"

#include "G4UIcmdWithoutParameter.hh"
#include "GateDetectorConstruction.hh"

//----------------------------------------------------------------------------------------
GateRunManagerMessenger::GateRunManagerMessenger(GateRunManager* runManager)
  : pRunManager(runManager)
{
  pRunInitCmd = new G4UIcmdWithoutParameter("/gate/run/initialize",this);
  pRunInitCmd->SetGuidance("Initialize geometry, actors and physics list.");

  pRunGeomUpdateCmd = new G4UIcmdWithoutParameter("/gate/geometry/rebuild",this);
  pRunGeomUpdateCmd->SetGuidance("Rebuild the whole geometry.");
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
GateRunManagerMessenger::~GateRunManagerMessenger()
{
  delete pRunInitCmd;
  delete pRunGeomUpdateCmd;
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
void GateRunManagerMessenger::SetNewValue(G4UIcommand* command, G4String /*newValue*/)
{   
  if (command == pRunInitCmd)
  {   
    pRunManager->InitializeAll();
  }
  else if (command == pRunGeomUpdateCmd)
  {
    pRunManager->InitGeometryOnly();
  }
}
//----------------------------------------------------------------------------------------
