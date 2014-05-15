/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*  Created: V. Cuplov   15 Feb. 2012
    Description: Geant4 Mie Scattering of Optical Photons based on Henyey-Greenstein phase function
*/

#include "GateOpticalMiePB.hh"
#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateOpticalMiePB::GateOpticalMiePB():GateVProcess("OpticalMie")
{  
  SetDefaultParticle("opticalphoton");
  SetProcessInfo("Mie process for optical photons");
  pMessenger = new GateEMStandardProcessMessenger(this) ;  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateOpticalMiePB::CreateProcess(G4ParticleDefinition *)
{

   return new G4OpMieHG(GetG4ProcessName());

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateOpticalMiePB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateOpticalMiePB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4OpticalPhoton::OpticalPhotonDefinition()) return true;
  return false;
}
//-----------------------------------------------------------------------------


MAKE_PROCESS_AUTO_CREATOR_CC(GateOpticalMiePB)
