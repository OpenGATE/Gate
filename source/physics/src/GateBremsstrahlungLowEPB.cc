/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef G4VERSION9_3
#include "GateBremsstrahlungLowEPB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateBremsstrahlungLowEPB::GateBremsstrahlungLowEPB():GateVProcess("LowEnergyBremsstrahlung")
{  
  SetDefaultParticle("e-");
  SetProcessInfo("Bremsstrahlung by electrons at low energy");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess *GateBremsstrahlungLowEPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4LowEnergyBremsstrahlung(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateBremsstrahlungLowEPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(GetProcess(),-1, -1, 3); //addProcessToManager(manager, -1,-1,3)          
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateBremsstrahlungLowEPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Electron::Electron() ) return true;
  return false;
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GateBremsstrahlungLowEPB)
#endif

