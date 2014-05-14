/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateEnergySpectrumActorMessenger.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateEnergySpectrumActor.hh"

//-----------------------------------------------------------------------------
GateEnergySpectrumActorMessenger::GateEnergySpectrumActorMessenger(GateEnergySpectrumActor * v)
: GateActorMessenger(v),
  pActor(v)
{

  BuildCommands(baseName+pActor->GetObjectName());

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateEnergySpectrumActorMessenger::~GateEnergySpectrumActorMessenger()
{
  delete pEmaxCmd;
  delete pEminCmd;
  delete pNBinsCmd;
  delete pEdepmaxCmd;
  delete pEdepminCmd;
  delete pEdepNBinsCmd;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateEnergySpectrumActorMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;

  bb = base+"/energySpectrum/setEmin";
  pEminCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set minimum energy of the energy spectrum");
  pEminCmd->SetGuidance(guidance);
  pEminCmd->SetParameterName("Emin", false);
  pEminCmd->SetDefaultUnit("MeV");

  bb = base+"/energySpectrum/setEmax";
  pEmaxCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set maximum energy of the energy spectrum");
  pEmaxCmd->SetGuidance(guidance);
  pEmaxCmd->SetParameterName("Emax", false);
  pEmaxCmd->SetDefaultUnit("MeV");

  bb = base+"/energySpectrum/setNumberOfBins";
  pNBinsCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = G4String("Set number of bins of the energy spectrum");
  pNBinsCmd->SetGuidance(guidance);
  pNBinsCmd->SetParameterName("Nbins", false);

  bb = base+"/energyLossHisto/setEmin";
  pEdepminCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set minimum energy of the energy loss histogram");
  pEdepminCmd->SetGuidance(guidance);
  pEdepminCmd->SetParameterName("Emin", false);
  pEdepminCmd->SetDefaultUnit("MeV");

  bb = base+"/energyLossHisto/setEmax";
  pEdepmaxCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set maximum energy of the energy loss histogram");
  pEdepmaxCmd->SetGuidance(guidance);
  pEdepmaxCmd->SetParameterName("Emax", false);
  pEdepmaxCmd->SetDefaultUnit("MeV");

  bb = base+"/energyLossHisto/setNumberOfBins";
  pEdepNBinsCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = G4String("Set number of bins of the energy loss histogram");
  pEdepNBinsCmd->SetGuidance(guidance);
  pEdepNBinsCmd->SetParameterName("Nbins", false);

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateEnergySpectrumActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{

  if(cmd == pEminCmd) pActor->SetEmin(  pEminCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pEmaxCmd) pActor->SetEmax(  pEmaxCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pNBinsCmd) pActor->SetENBins(  pNBinsCmd->GetNewIntValue(newValue)  ) ;
  if(cmd == pEdepminCmd) pActor->SetEdepmin(  pEdepminCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pEdepmaxCmd) pActor->SetEdepmax(  pEdepmaxCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pEdepNBinsCmd) pActor->SetEdepNBins(  pEdepNBinsCmd->GetNewIntValue(newValue)  ) ;

  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------

#endif
