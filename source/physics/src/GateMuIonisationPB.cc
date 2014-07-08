/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateMuIonisationPB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateMuIonisationPB::GateMuIonisationPB():GateVProcess("MuIonisation")
{  
  SetDefaultParticle("mu+"); SetDefaultParticle("mu-");


  SetProcessInfo("Ionization and energy loss by mu+ and mu-");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateMuIonisationPB::CreateProcess(G4ParticleDefinition *)
{
 return new G4MuIonisation(GetG4ProcessName()); 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMuIonisationPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(GetProcess(),-1, 2, 2);   
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateMuIonisationPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par->GetParticleName() != "mu-" && par->GetParticleName() != "mu+") return false;
  return true;
}
//-----------------------------------------------------------------------------



MAKE_PROCESS_AUTO_CREATOR_CC(GateMuIonisationPB)

