/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef G4VERSION9_3
#include "GatePhotoElectricLowEPB.hh"

#include "GatePhotoElectricMessenger.hh"

//-----------------------------------------------------------------------------
GatePhotoElectricLowEPB::GatePhotoElectricLowEPB():GateVProcess("LowEnergyPhotoElectric")
{  
  SetDefaultParticle("gamma");
  SetProcessInfo("Photo electric effect at low energy");
  pMessenger = new GatePhotoElectricMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GatePhotoElectricLowEPB::CreateProcess(G4ParticleDefinition *)
{
  G4LowEnergyPhotoElectric * proc = new G4LowEnergyPhotoElectric(GetG4ProcessName());

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
void GatePhotoElectricLowEPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePhotoElectricLowEPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GatePhotoElectricLowEPB)
#endif
