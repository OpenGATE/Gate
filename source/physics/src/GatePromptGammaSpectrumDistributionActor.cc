/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GatePromptGammaSpectrumDistributionActor.hh"
#include "GatePromptGammaSpectrumDistributionActorMessenger.hh"

#include <G4VProcess.hh>
#include <G4ProtonInelasticProcess.hh>
#include <G4CrossSectionDataStore.hh>

//-----------------------------------------------------------------------------
GatePromptGammaSpectrumDistributionActor::
GatePromptGammaSpectrumDistributionActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  DD("GPGSDA::Constructor");
  pMessenger = new GatePromptGammaSpectrumDistributionActorMessenger(this);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaSpectrumDistributionActor::~GatePromptGammaSpectrumDistributionActor()
{
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaSpectrumDistributionActor::Construct()
{
  DD("GPGSDA::Construct");
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
void GatePromptGammaSpectrumDistributionActor::SaveData()
{
  DD("GPGSDA::SaveData");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaSpectrumDistributionActor::ResetData()
{
  DD("GPGSDA::ResetData");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaSpectrumDistributionActor::PreUserTrackingAction(const GateVVolume*,
                                                                     const G4Track*)
{
  DD("GPGSDA::PreUserTrackingAction");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaSpectrumDistributionActor::UserSteppingAction(const GateVVolume*,
                                                                  const G4Step* step)
{
  DD("GPGSDA::UserSteppingAction");

  // Check particle type ("proton")
  const G4String particle_name = step->GetTrack()->GetParticleDefinition()->GetParticleName();
  if (particle_name != "proton") return;

  // Check process type name ("ProtonInelastic")

  // For all secondaries, store data

}
//-----------------------------------------------------------------------------
