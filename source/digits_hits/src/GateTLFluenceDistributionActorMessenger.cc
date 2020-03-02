/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateTLFluenceDistributionActorMessenger.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateTLFluenceDistributionActor.hh"


//-----------------------------------------------------------------------------
GateTLFluenceDistributionActorMessenger::GateTLFluenceDistributionActorMessenger(GateTLFluenceDistributionActor * v)
: GateActorMessenger(v),
  pActor(v)
{

  BuildCommands(baseName+pActor->GetObjectName());

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateTLFluenceDistributionActorMessenger::~GateTLFluenceDistributionActorMessenger()
{
  delete pEnergyMaxCmd;
  delete pEnergyMinCmd;
  delete pThetaMaxCmd;
  delete pThetaMinCmd;
  delete pPhiMaxCmd;
  delete pPhiMinCmd;
  
  delete pEnergyNBinsCmd;
  delete pThetaNBinsCmd;
  delete pPhiNBinsCmd;
  
  delete pEnergyEnableCmd;
  delete pThetaEnableCmd;
  delete pPhiEnableCmd;
  
  delete pAsciiFileCmd;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateTLFluenceDistributionActorMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;

  bb = base+"/setEnergyMin";
  pEnergyMinCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set minimum energy of the histogram");
  pEnergyMinCmd->SetGuidance(guidance);
  pEnergyMinCmd->SetParameterName("EnergyMin", false);
  pEnergyMinCmd->SetDefaultUnit("MeV");
  
  bb = base+"/setEnergyMax";
  pEnergyMaxCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set maximum energy of the histogram");
  pEnergyMaxCmd->SetGuidance(guidance);
  pEnergyMaxCmd->SetParameterName("EnergyMax", false);
  pEnergyMaxCmd->SetDefaultUnit("MeV");

  bb = base+"/setNEnergyBins";
  pEnergyNBinsCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = G4String("Set number of bins of energy the histogram");
  pEnergyNBinsCmd->SetGuidance(guidance);
  pEnergyNBinsCmd->SetParameterName("nEnergyBins", false);
  
  bb = base+"/enableEnergyHistogram";
  pEnergyEnableCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable the energy histogram");
  pEnergyEnableCmd->SetGuidance(guidance);
  pEnergyEnableCmd->SetParameterName("enableEnergyHistogram", false);

  
  bb = base+"/setThetaMin";
  pThetaMinCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set minimum angle of the histogram");
  pThetaMinCmd->SetGuidance(guidance);
  pThetaMinCmd->SetParameterName("ThetaMin", false);
  pThetaMinCmd->SetDefaultUnit("deg");
  
  bb = base+"/setThetaMax";
  pThetaMaxCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set maximum angle of the histogram");
  pThetaMaxCmd->SetGuidance(guidance);
  pThetaMaxCmd->SetParameterName("ThetaMax", false);
  pThetaMaxCmd->SetDefaultUnit("deg");

  bb = base+"/setNThetaBins";
  pThetaNBinsCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = G4String("Set number of bins of theta the histogram");
  pThetaNBinsCmd->SetGuidance(guidance);
  pThetaNBinsCmd->SetParameterName("nThetaBins", false);
  
  bb = base+"/enableThetaHistogram";
  pThetaEnableCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable the theta histogram");
  pThetaEnableCmd->SetGuidance(guidance);
  pThetaEnableCmd->SetParameterName("enableThetaHistogram", false);
  
  
  bb = base+"/setPhiMin";
  pPhiMinCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set minimum angle of the histogram");
  pPhiMinCmd->SetGuidance(guidance);
  pPhiMinCmd->SetParameterName("PhiMin", false);
  pPhiMinCmd->SetDefaultUnit("deg");
  
  bb = base+"/setPhiMax";
  pPhiMaxCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set maximum angle of the histogram");
  pPhiMaxCmd->SetGuidance(guidance);
  pPhiMaxCmd->SetParameterName("PhiMax", false);
  pPhiMaxCmd->SetDefaultUnit("deg");

  bb = base+"/setNPhiBins";
  pPhiNBinsCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = G4String("Set number of bins of theta the histogram");
  pPhiNBinsCmd->SetGuidance(guidance);
  pPhiNBinsCmd->SetParameterName("nPhiBins", false);
  
  bb = base+"/enablePhiHistogram";
  pPhiEnableCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable the phi histogram");
  pPhiEnableCmd->SetGuidance(guidance);
  pPhiEnableCmd->SetParameterName("enablePhiHistogram", false);
  
  
  bb = base+"/setAsciiFile";
  pAsciiFileCmd = new G4UIcmdWithAString(bb, this);
  guidance = G4String("Set ascii output file name");
  pAsciiFileCmd->SetGuidance(guidance);
  pAsciiFileCmd->SetParameterName("asciiFile", false);

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateTLFluenceDistributionActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if(cmd == pEnergyMaxCmd) pActor->SetEnergyMax(  pEnergyMaxCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pEnergyMinCmd) pActor->SetEnergyMin(  pEnergyMinCmd->GetNewDoubleValue(newValue)  ) ;
  // Angles are always in rad. Need to convert here
  if(cmd == pThetaMaxCmd)  pActor->SetThetaMax(180.0/CLHEP::pi * pThetaMaxCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pThetaMinCmd)  pActor->SetThetaMin(180.0/CLHEP::pi *  pThetaMinCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pPhiMaxCmd)    pActor->SetPhiMax(180.0/CLHEP::pi *  pPhiMaxCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pPhiMinCmd)    pActor->SetPhiMin(180.0/CLHEP::pi *  pPhiMinCmd->GetNewDoubleValue(newValue)  ) ;
  
  if(cmd == pEnergyNBinsCmd) pActor->SetNEnergyBins(  pEnergyNBinsCmd->GetNewIntValue(newValue)  ) ;
  if(cmd == pThetaNBinsCmd) pActor->SetNThetaBins(  pThetaNBinsCmd->GetNewIntValue(newValue)  ) ;
  if(cmd == pPhiNBinsCmd) pActor->SetNPhiBins(  pPhiNBinsCmd->GetNewIntValue(newValue)  ) ;

  
  if(cmd == pEnergyEnableCmd) pActor->enableEnergy(  pEnergyEnableCmd->GetNewBoolValue(newValue)  ) ;
  if(cmd == pThetaEnableCmd) pActor->enableTheta(  pThetaEnableCmd->GetNewBoolValue(newValue)  ) ;
  if(cmd == pPhiEnableCmd) pActor->enablePhi(  pPhiEnableCmd->GetNewBoolValue(newValue)  ) ;
  
  
  if(cmd == pAsciiFileCmd) pActor->SetAsciiFile( newValue ) ;

  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------

#endif
