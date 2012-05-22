/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#include "GateElectronIonisationLivermorePB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateElectronIonisationLivermorePB::GateElectronIonisationLivermorePB():GateVProcess("LivermoreElectronIonisation")
{  
  SetDefaultParticle("e-");
  SetProcessInfo("Ionization and energy loss by electrons at low energy");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateElectronIonisationLivermorePB::CreateProcess(G4ParticleDefinition * par)
{
  return dynamic_cast<G4VProcess*>( new G4LivermoreIonisationModel(par, GetG4ProcessName()) );
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateElectronIonisationLivermorePB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(GetProcess(),-1, 2, 2);           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateElectronIonisationLivermorePB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Electron::Electron()) return true;
  return false;
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GateElectronIonisationLivermorePB)

