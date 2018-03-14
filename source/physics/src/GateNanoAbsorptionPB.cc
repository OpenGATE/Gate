/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateNanoAbsorptionPB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateNanoAbsorptionPB::GateNanoAbsorptionPB():GateVProcess("NanoAbsorption")
{  
  SetDefaultParticle("opticalphoton");
  SetProcessInfo("Nano Particle Absorption process for optical photons");
  pMessenger = new GateEMStandardProcessMessenger(this) ;  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateNanoAbsorptionPB::CreateProcess(G4ParticleDefinition *)
{
  return new GateNanoAbsorption();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNanoAbsorptionPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateNanoAbsorptionPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4OpticalPhoton::OpticalPhotonDefinition()) return true;
  return false;
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GateNanoAbsorptionPB)
