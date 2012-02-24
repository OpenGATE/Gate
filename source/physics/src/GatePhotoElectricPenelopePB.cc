/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef G4VERSION9_3

#include "GatePhotoElectricPenelopePB.hh"

#include "GatePhotoElectricMessenger.hh"

//-----------------------------------------------------------------------------
GatePhotoElectricPenelopePB::GatePhotoElectricPenelopePB():GateVProcess("PenelopePhotoElectric")
{  
  SetDefaultParticle("gamma");
  SetProcessInfo("Photo electric effect (Penelope)");
  pMessenger = new GatePhotoElectricMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GatePhotoElectricPenelopePB::CreateProcess(G4ParticleDefinition *)
{
  G4PenelopePhotoElectric * proc = new G4PenelopePhotoElectric(GetG4ProcessName());

  bool auger = dynamic_cast<GatePhotoElectricMessenger*>(pMessenger)->GetIsAugerActivated();
  double lowEGamma = dynamic_cast<GatePhotoElectricMessenger*>(pMessenger)->GetLowEnergyGammaCut();
  double lowEElec = dynamic_cast<GatePhotoElectricMessenger*>(pMessenger)->GetLowEnergyElectronCut();
  if(auger) proc->ActivateAuger(true);
  if( lowEGamma>0 ) proc->SetCutForLowEnSecPhotons(lowEGamma);
  if( lowEElec>0  ) proc->SetCutForLowEnSecElectrons(lowEElec);

  return proc; 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhotoElectricPenelopePB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePhotoElectricPenelopePB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GatePhotoElectricPenelopePB)
#endif
