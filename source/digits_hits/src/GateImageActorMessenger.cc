/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEVIMAGEACTORMESSENGER_CC
#define GATEVIMAGEACTORMESSENGER_CC

#include "GateImageActorMessenger.hh"
#include "GateVImageActor.hh"

//#include "G4UIcmdWithABool.hh"

//-----------------------------------------------------------------------------
GateImageActorMessenger::GateImageActorMessenger(GateVImageActor * v)
: GateActorMessenger(v),
  pImageActor(v)
{

  BuildCommands(baseName+pImageActor->GetObjectName());

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateImageActorMessenger::~GateImageActorMessenger()
{
  delete pStepHitTypeCmd;
  delete pVoxelSizeCmd;
  delete pResolutionCmd;
  delete pHalfSizeCmd;
  delete pSizeCmd;
  delete pPositionCmd;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateImageActorMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;

  bb = base+"/setVoxelSize";
  pVoxelSizeCmd = new G4UIcmdWith3VectorAndUnit(bb, this);
  guidance = G4String("Sets voxel size");
  pVoxelSizeCmd->SetGuidance(guidance);
  pVoxelSizeCmd->SetParameterName("voxelsize_x","voxelsize_y", "voxelsize_z", false, false);
  pVoxelSizeCmd->SetDefaultUnit("mm");

  bb = base+"/setResolution";
  pResolutionCmd = new G4UIcmdWith3Vector(bb,this);
  guidance = G4String("Sets resolution");
  pResolutionCmd->SetGuidance(guidance);
  pResolutionCmd->SetParameterName("resolution_x","resolution_y", "resolution_z", false, false);

  bb = base+"/setHalfSize";
  pHalfSizeCmd = new G4UIcmdWith3VectorAndUnit(bb,this);
  guidance = G4String("Sets half size of the actor");
  pHalfSizeCmd->SetGuidance(guidance);
  pHalfSizeCmd->SetParameterName("halfsize_x","halfsize_y", "halfsize_z", false, false);
  pHalfSizeCmd->SetDefaultUnit("mm");

  bb = base+"/setSize";
  pSizeCmd = new G4UIcmdWith3VectorAndUnit(bb,this);
  guidance = G4String("Sets size of the actor");
  pSizeCmd->SetGuidance(guidance);
  pSizeCmd->SetParameterName("size_x","size_y", "size_z", false, false);
  pSizeCmd->SetDefaultUnit("mm");

  bb = base+"/setPosition";
  pPositionCmd = new G4UIcmdWith3VectorAndUnit(bb,this);
  guidance = G4String("Sets position (according to parent volume center)");
  pPositionCmd->SetGuidance(guidance);
  pPositionCmd->SetParameterName("position_x","position_y", "position_z", false, false);
  pPositionCmd->SetDefaultUnit("mm");

  bb = base +"/stepHitType";
  pStepHitTypeCmd = new G4UIcmdWithAString(bb,this);
  guidance = G4String("Sets  hit type ('pre', 'post', 'random' or 'middle'). Default is 'middle'.");
  pStepHitTypeCmd->SetGuidance(guidance);

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateImageActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue) {
  if (cmd == pVoxelSizeCmd)   pImageActor->SetVoxelSize(pVoxelSizeCmd->GetNew3VectorValue(newValue));
  if (cmd == pResolutionCmd)  pImageActor->SetResolution(pResolutionCmd->GetNew3VectorValue(newValue));
  if (cmd == pHalfSizeCmd)    pImageActor->SetHalfSize(pHalfSizeCmd->GetNew3VectorValue(newValue));
  if (cmd == pSizeCmd)        pImageActor->SetSize(pSizeCmd->GetNew3VectorValue(newValue));
  if (cmd == pPositionCmd)    pImageActor->SetPosition(pPositionCmd->GetNew3VectorValue(newValue));
  if (cmd == pStepHitTypeCmd) pImageActor->SetStepHitType(newValue);
  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEVIMAGEACTORMESSENGER_CC */
