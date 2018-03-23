/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateXrayBoundaryPB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateXrayBoundaryPB::GateXrayBoundaryPB():GateVProcess("XrayBoundary")
{
  SetDefaultParticle("gamma");
  SetProcessInfo("Boundary process for X-ray");
  pMessenger = new GateEMStandardProcessMessenger(this) ;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateXrayBoundaryPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4XrayBoundaryProcess(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateXrayBoundaryPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateXrayBoundaryPB::IsApplicable(G4ParticleDefinition * par)
{
  return ( par == G4Gamma::GammaDefinition() );
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GateXrayBoundaryPB)
