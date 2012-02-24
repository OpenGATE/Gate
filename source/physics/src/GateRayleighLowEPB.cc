/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef G4VERSION9_3
#include "GateRayleighLowEPB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateRayleighLowEPB::GateRayleighLowEPB():GateVProcess("LowEnergyRayleighScattering")
{  
  SetDefaultParticle("gamma");
  SetProcessInfo("Rayleigh scattering of gammas at low energy");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateRayleighLowEPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4LowEnergyRayleigh(GetG4ProcessName());
  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateRayleighLowEPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateRayleighLowEPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


MAKE_PROCESS_AUTO_CREATOR_CC(GateRayleighLowEPB)
#endif
