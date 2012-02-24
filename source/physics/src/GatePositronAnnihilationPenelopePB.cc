/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef G4VERSION9_3
#include "GatePositronAnnihilationPenelopePB.hh"


#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GatePositronAnnihilationPenelopePB::GatePositronAnnihilationPenelopePB():GateVProcess("PenelopePositronAnnihilation")
{  
  SetDefaultParticle("e+");
  SetProcessInfo("Positron annihilation (Penelope)");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GatePositronAnnihilationPenelopePB::CreateProcess(G4ParticleDefinition *)
{
  return new G4PenelopeAnnihilation();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePositronAnnihilationPenelopePB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(GetProcess(),0, -1, 4); 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePositronAnnihilationPenelopePB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Positron::Positron()) return true;
  return false;
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GatePositronAnnihilationPenelopePB)

#endif
