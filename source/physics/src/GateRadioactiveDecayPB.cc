/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateRadioactiveDecayPB.hh"

//User must first download radioactive decay file from Geant4 web page

//-----------------------------------------------------------------------------
GateRadioactiveDecayPB::GateRadioactiveDecayPB():GateVProcess("RadioactiveDecay")
{  
  SetDefaultParticle("GenericIon");


  SetProcessInfo("Decay of unstable particles");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateRadioactiveDecayPB::CreateProcess(G4ParticleDefinition *)
{
  //return new G4RadioactiveDecay(GetG4ProcessName());
  return new G4Radioactivation(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateRadioactiveDecayPB::ConstructProcess( G4ProcessManager * manager)
{  
//  manager->AddDiscreteProcess(GetProcess());
	  manager ->AddProcess(GetProcess());
	  manager ->SetProcessOrdering(GetProcess(), idxPostStep);
	  manager ->SetProcessOrdering(GetProcess(), idxAtRest);


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateRadioactiveDecayPB::IsApplicable(G4ParticleDefinition * /*par*/)
{
   return true; 
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GateRadioactiveDecayPB)
