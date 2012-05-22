/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#include "GateBremsstrahlungPenelopePB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateBremsstrahlungPenelopePB::GateBremsstrahlungPenelopePB():GateVProcess("PenelopeBremsstrahlung")
{  
  SetDefaultParticle("e+");
  SetDefaultParticle("e-");
  SetProcessInfo("Bremsstrahlung by electrons and positrons (Penelope)");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateBremsstrahlungPenelopePB::CreateProcess(G4ParticleDefinition * par)
{
  return dynamic_cast<G4VProcess*>( new G4PenelopeBremsstrahlungModel(par, GetG4ProcessName()) );
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateBremsstrahlungPenelopePB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(GetProcess(),-1, -1, 3);           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateBremsstrahlungPenelopePB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Electron::Electron() || par == G4Positron::Positron()) return true;
  return false;
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GateBremsstrahlungPenelopePB)

