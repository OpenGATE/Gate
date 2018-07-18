/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateProductionAndStoppingActorMessenger.hh"
#ifdef G4ANALYSIS_USE_ROOT

#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"

#include "GateProductionAndStoppingActor.hh"

//-----------------------------------------------------------------------------
GateProductionAndStoppingActorMessenger::GateProductionAndStoppingActorMessenger(GateProductionAndStoppingActor* sensor)
  :GateImageActorMessenger(sensor),pActor(sensor)
{
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateProductionAndStoppingActorMessenger::~GateProductionAndStoppingActorMessenger()
{
  delete bCoordinateFrameCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateProductionAndStoppingActorMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;

  bb = base+"/setCoordinateFrame";
  bCoordinateFrameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Store the hit coordinates in the frame of the volume passed as an argument.";
  bCoordinateFrameCmd->SetGuidance(guidance);
  bCoordinateFrameCmd->SetParameterName("Coordinate Frame",false);


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateProductionAndStoppingActorMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command == bCoordinateFrameCmd) {pActor->SetCoordFrame(param);pActor->SetEnableCoordFrame();};

  GateImageActorMessenger::SetNewValue(command ,param );
}
//-----------------------------------------------------------------------------

#endif
