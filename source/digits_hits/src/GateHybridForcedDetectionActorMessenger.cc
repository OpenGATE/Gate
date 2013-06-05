/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"
#ifdef GATE_USE_RTK

#ifndef GATEHYBRIDFORCEDDECTECTIONACTORMESSENGER_CC
#define GATEHYBRIDFORCEDDECTECTIONACTORMESSENGER_CC

#include "GateHybridForcedDetectionActorMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"

//-----------------------------------------------------------------------------
GateHybridForcedDetectionActorMessenger::GateHybridForcedDetectionActorMessenger(GateHybridForcedDetectionActor* sensor):
  GateActorMessenger(sensor),pHybridActor(sensor)
{
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateHybridForcedDetectionActorMessenger::~GateHybridForcedDetectionActorMessenger()
{
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHybridForcedDetectionActorMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;
 
  bb = base+"/setDetector";
  pSetDetectorCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the name of the volume used for detector (must be a Box).";
  pSetDetectorCmd->SetGuidance(guidance);

  bb = base+"/setDetectorResolution";
  pSetDetectorResolCmd = new GateUIcmdWith2Vector(bb, this);
  guidance = "Set the resolution of the detector (2D).";
  pSetDetectorResolCmd->SetGuidance(guidance);

  bb = base+"/geometryFilename";
  pSetGeometryFilenameCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Set the file name for the RTK geometry filename.";
  pSetGeometryFilenameCmd->SetGuidance(guidance);

  bb = base+"/primaryFilename";
  pSetPrimaryFilenameCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Set the file name for the primary x-rays (printf format with runId as a single parameter).";
  pSetPrimaryFilenameCmd->SetGuidance(guidance);

  bb = base+"/materialMuFilename";
  pSetMaterialMuFilenameCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Set the file name for writing the image that provides the attenuation of each material at each energy.";
  pSetMaterialMuFilenameCmd->SetGuidance(guidance);

  bb = base+"/attenuationFilename";
  pSetAttenuationFilenameCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Set the file name for writing the image that provides the attenuation image.";
  pSetAttenuationFilenameCmd->SetGuidance(guidance);

  bb = base+"/flatFieldFilename";
  pSetFlatFieldFilenameCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Set the file name for writing the image that provides the flat field image.";
  pSetFlatFieldFilenameCmd->SetGuidance(guidance);

  bb = base+"/comptonFilename";
  pSetComptonFilenameCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Set the file name for writing the image that provides the Compton image.";
  pSetComptonFilenameCmd->SetGuidance(guidance);

  bb = base+"/rayleighFilename";
  pSetRayleighFilenameCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Set the file name for writing the image that provides the rayleigh image.";
  pSetRayleighFilenameCmd->SetGuidance(guidance);

  bb = base+"/fluorescenceFilename";
  pSetFluorescenceFilenameCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Set the file name for writing the image that provides the fluorescence image.";
  pSetFluorescenceFilenameCmd->SetGuidance(guidance);
  
  bb = base+"/singleInteractionFilename";
  pSetSingleInteractionFilenameCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Set the file name for writing the result of a single interaction.";
  pSetSingleInteractionFilenameCmd->SetGuidance(guidance);

  bb = base+"/singleInteractionType";
  pSetSingleInteractionTypeCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the type of the single interaction (Compton or Rayleigh).";
  pSetSingleInteractionTypeCmd->SetGuidance(guidance);

  bb = base+"/singleInteractionPosition";
  pSetSingleInteractionPositionCmd = new G4UIcmdWith3VectorAndUnit(bb, this);
  guidance = "Set the position of the single interaction (3D).";
  pSetSingleInteractionPositionCmd->SetGuidance(guidance);

  bb = base+"/singleInteractionDirection";
  pSetSingleInteractionDirectionCmd = new G4UIcmdWith3Vector(bb, this);
  guidance = "Set the direction of the single interaction (3D).";
  pSetSingleInteractionDirectionCmd->SetGuidance(guidance);

  bb = base+"/singleInteractionEnergy";
  pSetSingleInteractionEnergyCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = "Set the energy of the single interaction.";
  pSetSingleInteractionEnergyCmd->SetGuidance(guidance);

  bb = base+"/singleInteractionZ";
  pSetSingleInteractionZCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = "Set the Z of the single interaction.";
  pSetSingleInteractionZCmd->SetGuidance(guidance);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHybridForcedDetectionActorMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command == pSetDetectorCmd) pHybridActor->SetDetectorVolumeName(param);
  if(command == pSetDetectorResolCmd) pHybridActor->SetDetectorResolution(pSetDetectorResolCmd->GetNew2VectorValue(param)[0], pSetDetectorResolCmd->GetNew2VectorValue(param)[1]);
  if(command == pSetGeometryFilenameCmd) pHybridActor->SetGeometryFilename(param);
  if(command == pSetPrimaryFilenameCmd) pHybridActor->SetPrimaryFilename(param);
  if(command == pSetMaterialMuFilenameCmd) pHybridActor->SetMaterialMuFilename(param);
  if(command == pSetAttenuationFilenameCmd) pHybridActor->SetAttenuationFilename(param);
  if(command == pSetFlatFieldFilenameCmd) pHybridActor->SetFlatFieldFilename(param);
  if(command == pSetComptonFilenameCmd) pHybridActor->SetComptonFilename(param);
  if(command == pSetRayleighFilenameCmd) pHybridActor->SetRayleighFilename(param);
  if(command == pSetFluorescenceFilenameCmd) pHybridActor->SetFluorescenceFilename(param);
  if(command == pSetSingleInteractionFilenameCmd) pHybridActor->SetSingleInteractionFilename(param);
  if(command == pSetSingleInteractionTypeCmd) pHybridActor->SetSingleInteractionType(param);
  if(command == pSetSingleInteractionPositionCmd) pHybridActor->SetSingleInteractionPosition(pSetSingleInteractionPositionCmd->GetNew3VectorValue(param));
  if(command == pSetSingleInteractionDirectionCmd) pHybridActor->SetSingleInteractionDirection(pSetSingleInteractionDirectionCmd->GetNew3VectorValue(param));
  if(command == pSetSingleInteractionEnergyCmd) pHybridActor->SetSingleInteractionEnergy(pSetSingleInteractionEnergyCmd->GetNewDoubleValue(param));
  if(command == pSetSingleInteractionZCmd) pHybridActor->SetSingleInteractionZ(pSetSingleInteractionZCmd->GetNewIntValue(param));

  GateActorMessenger::SetNewValue(command ,param );
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEHYBRIDFORCEDDECTECTIONACTORMESSENGER_CC */
#endif // GATE_USE_RTK
