/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GatePromptGammaSpectrumDistributionActorMessenger.hh"
#include "GatePromptGammaSpectrumDistributionActor.hh"

//-----------------------------------------------------------------------------
GatePromptGammaSpectrumDistributionActorMessenger::GatePromptGammaSpectrumDistributionActorMessenger(GatePromptGammaSpectrumDistributionActor* v)
  : GateActorMessenger(v), pActor(v)
{
  BuildCommands(baseName+pActor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaSpectrumDistributionActorMessenger::~GatePromptGammaSpectrumDistributionActorMessenger()
{

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaSpectrumDistributionActorMessenger::BuildCommands(G4String /*base*/)
{
  //bb = base+"/energySpectrum/setEmin";
  //pEminCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  //guidance = G4String("Set minimum energy of the energy spectrum");
  //pEminCmd->SetGuidance(guidance);
  //pEminCmd->SetParameterName("Emin", false);
  //pEminCmd->SetDefaultUnit("MeV");

  //bb = base+"/energyLossHisto/setNumberOfBins";
  //pEdepNBinsCmd = new G4UIcmdWithAnInteger(bb, this);
  //guidance = G4String("Set number of bins of the energy loss histogram");
  //pEdepNBinsCmd->SetGuidance(guidance);
  //pEdepNBinsCmd->SetParameterName("Nbins", false);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaSpectrumDistributionActorMessenger::SetNewValue(G4UIcommand* cmd,
                                                                    G4String newValue)
{
  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------
