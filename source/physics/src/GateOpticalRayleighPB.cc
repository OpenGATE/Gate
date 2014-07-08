/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateOpticalRayleighPB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateOpticalRayleighPB::GateOpticalRayleighPB():GateVProcess("OpticalRayleigh")
{  
  SetDefaultParticle("opticalphoton");
  SetProcessInfo("Rayleigh process for optical photons");
  pMessenger = new GateEMStandardProcessMessenger(this) ;  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateOpticalRayleighPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4OpRayleigh(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateOpticalRayleighPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateOpticalRayleighPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4OpticalPhoton::OpticalPhotonDefinition()) return true;
  return false;
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GateOpticalRayleighPB)
