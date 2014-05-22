/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*  Created: V. Cuplov   7 March 2012
    Description: Geant4 WaveLength Shifting (i.e. Fluorescence)
*/

#include "GateOpticalWLSPB.hh"
#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateOpticalWLSPB::GateOpticalWLSPB():GateVProcess("OpticalWLS")
{  
  SetDefaultParticle("opticalphoton");
  SetProcessInfo("Wave Length Shifting process for optical photons");
  pMessenger = new GateEMStandardProcessMessenger(this) ;  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateOpticalWLSPB::CreateProcess(G4ParticleDefinition *)
{

   return new G4OpWLS(GetG4ProcessName());

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateOpticalWLSPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateOpticalWLSPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4OpticalPhoton::OpticalPhotonDefinition()) return true;
  return false;
}
//-----------------------------------------------------------------------------


MAKE_PROCESS_AUTO_CREATOR_CC(GateOpticalWLSPB)
