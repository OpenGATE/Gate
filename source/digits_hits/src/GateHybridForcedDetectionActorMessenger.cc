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

  // bb = base+"/setParticleName";
  // pSetParticleNameCmd = new G4UIcmdWithAString(bb,this);
  // guidance = "Set the particle name for the calculation.";
  // pSetParticleNameCmd->SetGuidance(guidance);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHybridForcedDetectionActorMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command == pSetDetectorCmd) pHybridActor->SetDetectorVolumeName(param);
  if(command == pSetDetectorResolCmd) pHybridActor->SetDetectorResolution(pSetDetectorResolCmd->GetNew2VectorValue(param)[0], pSetDetectorResolCmd->GetNew2VectorValue(param)[1]);
  
  GateActorMessenger::SetNewValue(command ,param );}
//-----------------------------------------------------------------------------

#endif /* end #define GATEHYBRIDFORCEDDECTECTIONACTORMESSENGER_CC */
#endif // GATE_USE_RTK
