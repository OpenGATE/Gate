/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateMuPairProductionPB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateMuPairProductionPB::GateMuPairProductionPB():GateVProcess("MuPairProduction")
{  
  SetDefaultParticle("mu+");
  SetDefaultParticle("mu-");
  SetProcessInfo("Direct production of (e+, e-) pairs by mu+ and mu-");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateMuPairProductionPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4MuPairProduction(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMuPairProductionPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateMuPairProductionPB::IsApplicable(G4ParticleDefinition * par)
{
 if(par->GetParticleName() != "mu-" && par->GetParticleName() != "mu+") return false;
  return true;
}
//-----------------------------------------------------------------------------


MAKE_PROCESS_AUTO_CREATOR_CC(GateMuPairProductionPB)
