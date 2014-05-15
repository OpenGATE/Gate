/*##########################################
#developed by Hermann Fuchs
#
#Christian Doppler Laboratory for Medical Radiation Research for Radiation Oncology
#Department of Radiation Oncology
#Medical University of Vienna
#
#and 
#
#Pierre Gueth
#CREATIS
#
#July 2012
##########################################
*/
/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCerenkovPB.hh"
#include "G4Cerenkov.hh"

#include "GateCerenkovMessenger.hh"


//-----------------------------------------------------------------------------
GateCerenkovPB::GateCerenkovPB():GateVProcess("Cerenkov")
{  
  SetDefaultParticle("e+"); SetDefaultParticle("e-");
  SetDefaultParticle("mu+"); SetDefaultParticle("mu-");
  SetDefaultParticle("tau+"); SetDefaultParticle("tau-");
  SetDefaultParticle("pi+"); SetDefaultParticle("pi-");
  SetDefaultParticle("kaon+"); SetDefaultParticle("kaon-");
  SetDefaultParticle("sigma+"); SetDefaultParticle("sigma-");
  SetDefaultParticle("proton"); SetDefaultParticle("anti_proton");
  SetDefaultParticle("xi-"); SetDefaultParticle("anti_xi-");
  SetDefaultParticle("anti_sigma+"); SetDefaultParticle("anti_sigma-");
  SetDefaultParticle("omega-"); SetDefaultParticle("anti_omega-");
  SetDefaultParticle("deuteron");
  SetDefaultParticle("triton");
  SetDefaultParticle("He3");
  SetDefaultParticle("alpha");
  SetDefaultParticle("GenericIon");
  
  SetProcessInfo("Cerenkov effect");

  pMessenger = new GateCerenkovMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateCerenkovPB::CreateProcess(G4ParticleDefinition *)
{
	G4Cerenkov* cerenkov = new G4Cerenkov(GetG4ProcessName());
	//G4cout << "creating " << cerenkov << G4endl;
	GateCerenkovMessenger* messenger = static_cast<GateCerenkovMessenger*>(pMessenger);
    cerenkov->SetMaxNumPhotonsPerStep(messenger->GetMaxNumPhotonsPerStep());
    cerenkov->SetTrackSecondariesFirst(messenger->GetTrackSecondariesFirst());
	cerenkov->SetMaxBetaChangePerStep(messenger->GetMaxBetaChangePerStep());
	return cerenkov;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCerenkovPB::ConstructProcess(G4ProcessManager * manager)
{
	//G4cout << "contructing " << GetProcess() << G4endl;
	manager->AddProcess(GetProcess());
	manager->SetProcessOrdering(GetProcess(), idxPostStep);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateCerenkovPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par->GetPDGCharge()!=0) return true;
  return false;
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GateCerenkovPB)
