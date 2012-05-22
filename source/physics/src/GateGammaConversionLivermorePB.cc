/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#include "GateGammaConversionLivermorePB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateGammaConversionLivermorePB::GateGammaConversionLivermorePB():GateVProcess("LivermoreGammaConversion")
{  
  SetDefaultParticle("gamma");
  SetProcessInfo("Pair production by gammas at low energy");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateGammaConversionLivermorePB::CreateProcess(G4ParticleDefinition * par)
{
  return dynamic_cast<G4VProcess*>( new G4LivermoreGammaConversionModel(par, GetG4ProcessName()) );
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateGammaConversionLivermorePB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateGammaConversionLivermorePB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


MAKE_PROCESS_AUTO_CREATOR_CC(GateGammaConversionLivermorePB)
