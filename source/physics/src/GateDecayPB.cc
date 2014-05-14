/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateDecayPB.hh"

//Even though the decay process may be assigned to neutrons, anti_neutrons, mu+ and mu-, these particles are treated as stable by G4Decay.
//To make a particle unstable (with the correct lifetime) use the method SetPDGStable(false) of the particle.

//-----------------------------------------------------------------------------
GateDecayPB::GateDecayPB():GateVProcess("Decay")
{  
  SetDefaultParticle("mu+");  SetDefaultParticle("mu-");
  SetDefaultParticle("tau+");  SetDefaultParticle("tau-");
  SetDefaultParticle("sigma0");  SetDefaultParticle("anti_sigma0");
  SetDefaultParticle("pi+"); SetDefaultParticle("pi-");SetDefaultParticle("pi0");
  SetDefaultParticle("kaon+"); SetDefaultParticle("kaon-");
  SetDefaultParticle("kaon0L");SetDefaultParticle("kaon0S");
  SetDefaultParticle("sigma+"); SetDefaultParticle("sigma-");
  SetDefaultParticle("neutron");  //SetDefaultParticle("anti_neutron"); ??? Error message:  "Decay is not applicable to anti_neutron"
  SetDefaultParticle("lambda");SetDefaultParticle("anti_lambda");
  SetDefaultParticle("xi-"); SetDefaultParticle("anti_xi-");
  SetDefaultParticle("anti_sigma+"); SetDefaultParticle("anti_sigma-");
  SetDefaultParticle("omega-"); SetDefaultParticle("anti_omega-");
  SetDefaultParticle("xi0");SetDefaultParticle("anti_xi0");


  SetProcessInfo("Decay of unstable particles");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateDecayPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4Decay(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDecayPB::ConstructProcess( G4ProcessManager * manager)
{  
  manager->AddProcess(GetProcess());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateDecayPB::IsApplicable(G4ParticleDefinition * par)
{
   // check if the particle is stable?
   if (par->GetPDGLifeTime() <0.0) {
     return false;
   } else if (par->GetPDGMass() <= 0.0*MeV) {
     return false;
   } else {
     return true; 
   }
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GateDecayPB)
