/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateMuBremsstrahlungPB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateMuBremsstrahlungPB::GateMuBremsstrahlungPB():GateVProcess("MuBremsstrahlung")
{  
  SetDefaultParticle("mu+");
  SetDefaultParticle("mu-");
  SetProcessInfo("Bremsstrahlung by mu- and mu+");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateMuBremsstrahlungPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4MuBremsstrahlung(GetG4ProcessName());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMuBremsstrahlungPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(pFinalProcess,-1, -1, 3);           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateMuBremsstrahlungPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par->GetParticleName() != "mu-" && par->GetParticleName() != "mu+") return false;
  return true;
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GateMuBremsstrahlungPB)

