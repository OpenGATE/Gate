/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSingleScatteringPB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateSingleScatteringPB::GateSingleScatteringPB():GateVProcess("SingleScattering")
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
  SetProcessInfo("Single Coulomb scattering of charged particles");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateSingleScatteringPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4CoulombScattering(GetG4ProcessName()); 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSingleScatteringPB::ConstructProcess(G4ProcessManager * manager)
{
  //manager->AddProcess(GetProcess(),-1, 1, 1);   
  manager->AddDiscreteProcess(GetProcess());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateSingleScatteringPB::IsApplicable(G4ParticleDefinition * par)
{
  for(unsigned int i=0; i<theListOfDefaultParticles.size(); i++)
      if(par->GetParticleName() == theListOfDefaultParticles[i]) return true;
  return false;
}
//-----------------------------------------------------------------------------


MAKE_PROCESS_AUTO_CREATOR_CC(GateSingleScatteringPB)
