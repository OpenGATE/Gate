/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateScintillationPB.hh"

#include "GateScintillation.hh"
#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateScintillationPB::GateScintillationPB():GateVProcess("Scintillation")
{  
  SetDefaultParticle("e+"); SetDefaultParticle("e-");
  SetDefaultParticle("mu+"); SetDefaultParticle("mu-");
  SetDefaultParticle("tau+"); SetDefaultParticle("tau-");
  SetDefaultParticle("pi+"); SetDefaultParticle("pi-");
  SetDefaultParticle("kaon+"); SetDefaultParticle("kaon-");
  SetDefaultParticle("sigma+"); SetDefaultParticle("sigma-");
  SetDefaultParticle("proton"); SetDefaultParticle("anti_proton");
  SetDefaultParticle("xi-"); SetDefaultParticle("anti_xi-");
  SetDefaultParticle("anti_sigma+"); SetDefaultParticle("anti_sigma-");
  SetDefaultParticle("omega-"); SetDefaultParticle("anti_omega-");
  SetDefaultParticle("deuteron");
  SetDefaultParticle("triton");
  SetDefaultParticle("He3");
  SetDefaultParticle("alpha");
  SetDefaultParticle("GenericIon");
  SetDefaultParticle("neutron");
  SetDefaultParticle("gamma");

  SetProcessInfo("Scintillation process");

  pMessenger = new GateEMStandardProcessMessenger(this) ;  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateScintillationPB::CreateProcess(G4ParticleDefinition *)
{
  GateScintillation* scintillation = new GateScintillation(GetG4ProcessName());
  scintillation->SetTrackSecondariesFirst(true);
  scintillation->SetScintillationYieldFactor(1);
  scintillation->SetScintillationExcitationRatio(0.0);

  return scintillation;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateScintillationPB::ConstructProcess(G4ProcessManager * manager)
{

  auto ret = manager->AddDiscreteProcess(GetProcess());
  if(ret < 0)
  {
    GateError("Can not add scintillation to particle '" << manager->GetParticleType()->GetParticleName() << "'."   );
  }
  manager->SetProcessOrderingToLast(GetProcess(), idxAtRest);
  manager->SetProcessOrderingToLast(GetProcess(), idxPostStep);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateScintillationPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par != G4OpticalPhoton::OpticalPhotonDefinition()) return true;
  return false;
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GateScintillationPB)
