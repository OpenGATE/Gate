/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GatePromptGammaProductionTLEActor.hh"
#include "GatePromptGammaProductionTLEActorMessenger.hh"

//-----------------------------------------------------------------------------
GatePromptGammaProductionTLEActor::GatePromptGammaProductionTLEActor(G4String name,
                                                                     G4int depth):
  GateVActor(name,depth)
{
  DD("GPGPTLE::Constructor");
  pMessenger = new GatePromptGammaProductionTLEActorMessenger(this);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaProductionTLEActor::~GatePromptGammaProductionTLEActor()
{
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaProductionTLEActor::Construct()
{
  DD("GPGPTLE::Construct");
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(false);
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);

  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaProductionTLEActor::ResetData()
{
  DD("GPGPTLE::ResetData");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaProductionTLEActor::SaveData()
{
  DD("GPGPTLE::SaveData");

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaProductionTLEActor::PreUserTrackingAction(const GateVVolume*, const G4Track*)
{
  DD("GPGPTLE::PreUserTrackingAction");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaProductionTLEActor::UserSteppingAction(const GateVVolume*,
                                                           const G4Step* step)
{
  DD("GPGPTLE::UserSteppingAction");

  // Check particle type ("proton")
  const G4String particle_name = step->GetTrack()->GetParticleDefinition()->GetParticleName();
  if (particle_name != "proton") return;


}
//-----------------------------------------------------------------------------
