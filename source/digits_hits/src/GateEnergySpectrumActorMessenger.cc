/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateEnergySpectrumActorMessenger.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateEnergySpectrumActor.hh"
#include "G4SystemOfUnits.hh"

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
  delete pLETminCmd;
  delete pLETmaxCmd;
  delete pNLETBinsCmd;
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

  bb = base+"/saveAsText";
  pSaveAsText = new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition to root output files, also write .txt files (that can be open as a source, 'UserSpectrum')");
  pSaveAsText->SetGuidance(guidance);

  bb = base+"/saveAsTextDiscreteEnergySpectrum";
  pSaveAsTextDiscreteEnergySpectrum = new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition to root output files, also write .txt files, discrete spectrum (that can be open as a source, 'UserSpectrum')");
  pSaveAsTextDiscreteEnergySpectrum->SetGuidance(guidance);
  
  
  bb = base+"/enableLETSpectrum";
  pEnableLETSpectrumCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable LET spectrum");
  pEnableLETSpectrumCmd->SetGuidance(guidance);

  bb = base+"/LETSpectrum/setLETmin";
  pLETminCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set minimum LET of the LET spectrum");
  pLETminCmd->SetGuidance(guidance);
  pLETminCmd->SetParameterName("LETmin", false);
  pLETminCmd->SetDefaultUnit("keV/um");
  
  bb = base+"/LETSpectrum/setLETmax";
  pLETmaxCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set maximum LET of the LET spectrum");
  pLETmaxCmd->SetGuidance(guidance);
  pLETmaxCmd->SetParameterName("LETmax", false);
  pLETmaxCmd->SetDefaultUnit("keV/um");
  
  bb = base+"/LETSpectrum/setNumberOfBins";
  pNLETBinsCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = G4String("Set number of bins of the energy spectrum");
  pNLETBinsCmd->SetGuidance(guidance);
  pNLETBinsCmd->SetParameterName("NLETbins", false);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEnergySpectrumActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if(cmd == pEminCmd) pActor->SetEmin(  pEminCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pEmaxCmd) pActor->SetEmax(  pEmaxCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pLETminCmd){
	   pActor->SetLETmin(  pLETminCmd->GetNewDoubleValue(newValue)  ) ;
	   //G4cout<<"Yes this is LETmin set " << pLETminCmd->GetNewDoubleValue(newValue)<<G4endl<<G4endl<<G4endl<<G4endl;
	   //G4cout<<"Wirklich"<<G4endl;
   }
  if(cmd == pLETmaxCmd) pActor->SetLETmax(  pLETmaxCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pNLETBinsCmd) pActor->SetNLETBins(  pNLETBinsCmd->GetNewIntValue(newValue)  ) ;
  
  if(cmd == pNBinsCmd) pActor->SetENBins(  pNBinsCmd->GetNewIntValue(newValue)  ) ;
  if(cmd == pEdepminCmd) pActor->SetEdepmin(  pEdepminCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pEdepmaxCmd) pActor->SetEdepmax(  pEdepmaxCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pEdepNBinsCmd) pActor->SetEdepNBins(  pEdepNBinsCmd->GetNewIntValue(newValue)  ) ;
  if(cmd == pSaveAsText) pActor->SetSaveAsTextFlag(  pSaveAsText->GetNewBoolValue(newValue)  ) ;
  if(cmd == pSaveAsTextDiscreteEnergySpectrum) pActor->SetSaveAsTextDiscreteEnergySpectrumFlag(  pSaveAsTextDiscreteEnergySpectrum->GetNewBoolValue(newValue)  ) ;
  if(cmd == pEnableLETSpectrumCmd) pActor->SetLETSpectrumCalc(  pEnableLETSpectrumCmd->GetNewBoolValue(newValue)  ) ;
  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------

#endif
