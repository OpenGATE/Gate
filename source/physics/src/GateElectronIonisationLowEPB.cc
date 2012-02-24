/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef G4VERSION9_3
#include "GateElectronIonisationLowEPB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateElectronIonisationLowEPB::GateElectronIonisationLowEPB():GateVProcess("LowEnergyElectronIonisation")
{  
  SetDefaultParticle("e-");
  SetProcessInfo("Ionization and energy loss by electrons at low energy");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateElectronIonisationLowEPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4LowEnergyIonisation(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateElectronIonisationLowEPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(GetProcess(),-1, 2, 2);           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateElectronIonisationLowEPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Electron::Electron()) return true;
  return false;
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GateElectronIonisationLowEPB)
#endif

