/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#include "GateBremsstrahlungLivermorePB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateBremsstrahlungLivermorePB::GateBremsstrahlungLivermorePB():GateVProcess("LivermoreBremsstrahlung")
{  
  SetDefaultParticle("e-");
  SetProcessInfo("Bremsstrahlung by electrons at low energy");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess *GateBremsstrahlungLivermorePB::CreateProcess(G4ParticleDefinition * par )
{
  return dynamic_cast<G4VProcess*>( new G4LivermoreBremsstrahlungModel(par,GetG4ProcessName()) );
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateBremsstrahlungLivermorePB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(GetProcess(),-1, -1, 3); //addProcessToManager(manager, -1,-1,3)          
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateBremsstrahlungLivermorePB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Electron::Electron() ) return true;
  return false;
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GateBremsstrahlungLivermorePB)

