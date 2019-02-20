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
  delete pQminCmd;
  delete pQmaxCmd;
  delete pNQBinsCmd;
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
  
  
  bb = base+"/enableQSpectrum";
  pEnableQSpectrumCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable Q spectrum");
  pEnableQSpectrumCmd->SetGuidance(guidance);
  
    bb = base+"/QSpectrum/setQmin";
  pQminCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set minimum Q of the Q spectrum");
  pQminCmd->SetGuidance(guidance);
  pQminCmd->SetParameterName("Qmin", false);
  pQminCmd->SetDefaultUnit("keV/um");
  
  bb = base+"/QSpectrum/setQmax";
  pQmaxCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set maximum Q of the Q spectrum");
  pQmaxCmd->SetGuidance(guidance);
  pQmaxCmd->SetParameterName("Qmax", false);
  pQmaxCmd->SetDefaultUnit("keV/um");
  
  bb = base+"/QSpectrum/setNumberOfBins";
  pNQBinsCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = G4String("Set number of bins of the energy spectrum");
  pNQBinsCmd->SetGuidance(guidance);
  pNQBinsCmd->SetParameterName("NQbins", false);
  
  
  bb = base+"/enableNbPartSpectrum";
  pEnableEnergySpectrumNbPartCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable Number of Particle spectrum");
  pEnableEnergySpectrumNbPartCmd->SetGuidance(guidance);
  
  bb = base+"/enableFluenceCosSpectrum";
  pEnableEnergySpectrumFluenceCosCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable fluence spectrum from momentum direction");
  pEnableEnergySpectrumFluenceCosCmd->SetGuidance(guidance);
  
  bb = base+"/enableFluenceTrackSpectrum";
  pEnableEnergySpectrumFluenceTrackCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable fluence spectrum from track length in volume");
  pEnableEnergySpectrumFluenceTrackCmd->SetGuidance(guidance);
  
  bb = base+"/enableEdepSpectrum";
  pEnableeEnergySpectrumEdepCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable energy deposition spectrum");
  pEnableeEnergySpectrumEdepCmd->SetGuidance(guidance);
  
  bb = base+"/enableEdepHisto";
  pEnableEdepHistoCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable edep histogram");
  pEnableEdepHistoCmd->SetGuidance(guidance);
  
  bb = base+"/enableEdepTimeHisto";
  pEnableEdepTimeHistoCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable edep time histogram");
  pEnableEdepTimeHistoCmd->SetGuidance(guidance);
  
  bb = base+"/enableEdepTrackHisto";
  pEnableEdepTrackHistoCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable edep track histogram");
  pEnableEdepTrackHistoCmd->SetGuidance(guidance);
  
  bb = base+"/enableElossHisto";
  pEnableElossHistoCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable energy loss histogram");
  pEnableElossHistoCmd->SetGuidance(guidance);
  
  bb = base+"/setLogBinWidth";
  pEnableLogBinningCMD = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Set logarithmic binning in energy");
  pEnableLogBinningCMD->SetGuidance(guidance);
  
  bb = base+"/setEnergyPerUnitMass";
  pEnableEnergyPerUnitMassCMD = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Set energy per nucleus");
  pEnableEnergyPerUnitMassCMD->SetGuidance(guidance);
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
  
  if(cmd == pQminCmd){  pActor->SetQmin(  pQminCmd->GetNewDoubleValue(newValue)  ) ;   }
  if(cmd == pQmaxCmd) pActor->SetQmax(  pQmaxCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pNQBinsCmd) pActor->SetNQBins(  pNQBinsCmd->GetNewIntValue(newValue)  ) ;
  
  if(cmd == pNBinsCmd) pActor->SetENBins(  pNBinsCmd->GetNewIntValue(newValue)  ) ;
  if(cmd == pEdepminCmd) pActor->SetEdepmin(  pEdepminCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pEdepmaxCmd) pActor->SetEdepmax(  pEdepmaxCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pEdepNBinsCmd) pActor->SetEdepNBins(  pEdepNBinsCmd->GetNewIntValue(newValue)  ) ;
  if(cmd == pSaveAsText) pActor->SetSaveAsTextFlag(  pSaveAsText->GetNewBoolValue(newValue)  ) ;
  if(cmd == pSaveAsTextDiscreteEnergySpectrum) pActor->SetSaveAsTextDiscreteEnergySpectrumFlag(  pSaveAsTextDiscreteEnergySpectrum->GetNewBoolValue(newValue)  ) ;
  if(cmd == pEnableLETSpectrumCmd) pActor->SetLETSpectrumCalc(  pEnableLETSpectrumCmd->GetNewBoolValue(newValue)  ) ;
  if(cmd == pEnableQSpectrumCmd) pActor->SetQSpectrumCalc(  pEnableQSpectrumCmd->GetNewBoolValue(newValue)  ) ;
  
  if(cmd == pEnableEnergySpectrumNbPartCmd) pActor->SetESpectrumNbPartCalc(  pEnableEnergySpectrumNbPartCmd->GetNewBoolValue(newValue)  ) ;
  if(cmd == pEnableEnergySpectrumFluenceCosCmd) pActor->SetESpectrumFluenceCosCalc(  pEnableEnergySpectrumFluenceCosCmd->GetNewBoolValue(newValue)  ) ;
  if(cmd == pEnableEnergySpectrumFluenceTrackCmd) pActor->SetESpectrumFluenceTrackCalc(  pEnableEnergySpectrumFluenceTrackCmd->GetNewBoolValue(newValue)  ) ;
  
  if(cmd == pEnableeEnergySpectrumEdepCmd) pActor->SetESpectrumEdepCalc(  pEnableeEnergySpectrumEdepCmd->GetNewBoolValue(newValue)  ) ;
  if(cmd == pEnableEdepHistoCmd) pActor->SetEdepHistoCalc (  pEnableEdepHistoCmd->GetNewBoolValue(newValue)  ) ;
  if(cmd == pEnableEdepTimeHistoCmd) pActor->SetEdepTimeHistoCalc(  pEnableEdepTimeHistoCmd->GetNewBoolValue(newValue)  ) ;
  if(cmd == pEnableEdepTrackHistoCmd) pActor->SetEdepTrackHistoCalc(  pEnableEdepTrackHistoCmd->GetNewBoolValue(newValue)  ) ;
  if(cmd == pEnableElossHistoCmd) pActor->SetElossHistoCalc(  pEnableElossHistoCmd->GetNewBoolValue(newValue)  ) ;
  
  if(cmd == pEnableLogBinningCMD) pActor->SetLogBinning(  pEnableLogBinningCMD->GetNewBoolValue(newValue)  ) ;
  if(cmd == pEnableEnergyPerUnitMassCMD) pActor->SetEnergyPerUnitMass(  pEnableEnergyPerUnitMassCMD->GetNewBoolValue(newValue)  ) ;
  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------

#endif
