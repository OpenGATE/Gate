/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef G4VERSION9_3
#include "GateGammaConversionLowEPB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateGammaConversionLowEPB::GateGammaConversionLowEPB():GateVProcess("LowEnergyGammaConversion")
{  
  SetDefaultParticle("gamma");
  SetProcessInfo("Pair production by gammas at low energy");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateGammaConversionLowEPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4LowEnergyGammaConversion(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateGammaConversionLowEPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateGammaConversionLowEPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


MAKE_PROCESS_AUTO_CREATOR_CC(GateGammaConversionLowEPB)
#endif
