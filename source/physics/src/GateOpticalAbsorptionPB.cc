/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateOpticalAbsorptionPB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateOpticalAbsorptionPB::GateOpticalAbsorptionPB():GateVProcess("OpticalAbsorption")
{  
  SetDefaultParticle("opticalphoton");
  SetProcessInfo("Absorption process for optical photons");
  pMessenger = new GateEMStandardProcessMessenger(this) ;  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateOpticalAbsorptionPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4OpAbsorption(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateOpticalAbsorptionPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateOpticalAbsorptionPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4OpticalPhoton::OpticalPhotonDefinition()) return true;
  return false;
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GateOpticalAbsorptionPB)
