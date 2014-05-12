/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GatePositronAnnihilationPB.hh"
#include "GateEMStandardProcessMessenger.hh"


//-----------------------------------------------------------------------------
GatePositronAnnihilationPB::GatePositronAnnihilationPB():GateVProcess("PositronAnnihilation")
{  
  SetDefaultParticle("e+");
  SetProcessInfo("Positron annihilation with accolinearity");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GatePositronAnnihilationPB::CreateProcess(G4ParticleDefinition *)
{
  return new GatePositronAnnihilation(); 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePositronAnnihilationPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(GetProcess(),0, -1, 4); 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePositronAnnihilationPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Positron::Positron()) return true;
  return false;
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GatePositronAnnihilationPB)

