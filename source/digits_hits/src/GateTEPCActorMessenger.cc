/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateTEPCActorMessenger.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateTEPCActor.hh"

//-----------------------------------------------------------------------------
GateTEPCActorMessenger::GateTEPCActorMessenger(GateTEPCActor * v)
  : GateActorMessenger(v),
    pActor(v)
{
  BuildCommands(baseName+pActor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateTEPCActorMessenger::~GateTEPCActorMessenger()
{
  delete pPressureCmd;
  delete pEmaxCmd;
  delete pEminCmd;
  delete pEBinNumberCmd;
  delete pELogscaleCmd;
  delete pENOrdersCmd;
  delete pNormByEventCmd;
  delete pSaveAsTextCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTEPCActorMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;

  bb = base+"/setPressure";
  pPressureCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set the pressure of the TEPC gas (G4_TISSUE-PROPANE by default)");
  pPressureCmd->SetGuidance(guidance);
  pPressureCmd->SetParameterName("Pressure", false);
  
  bb = base+"/setEmin";
  pEminCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set minimum energy of the LET spectrum");
  pEminCmd->SetGuidance(guidance);
  pEminCmd->SetParameterName("Emin", false);
  pEminCmd->SetDefaultUnit("MeV");

  bb = base+"/setEmax";
  pEmaxCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set maximum energy of the LET spectrum");
  pEmaxCmd->SetGuidance(guidance);
  pEmaxCmd->SetParameterName("Emax", false);
  pEmaxCmd->SetDefaultUnit("MeV");

  bb = base+"/setNumberOfBins";
  pEBinNumberCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = G4String("Set number of bins of the LET spectrum");
  pEBinNumberCmd->SetGuidance(guidance);
  pEBinNumberCmd->SetParameterName("Nbins", false);
  
  bb = base+"/setLogscale";
  pELogscaleCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("set the bin size according to a log scale");
  pELogscaleCmd->SetGuidance(guidance);
  pELogscaleCmd->SetParameterName("ELogscale", false);
  
  bb = base+"/setNOrders";
  pENOrdersCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = G4String("when using the logscale, set the number of orders from Emin");
  pENOrdersCmd->SetGuidance(guidance);
  pENOrdersCmd->SetParameterName("NOrders", false);
  
  bb = base+"/setNormByEvent";
  pNormByEventCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("normalize LET spectrum by the number of event");
  pNormByEventCmd->SetGuidance(guidance);
  pNormByEventCmd->SetParameterName("NormByEvent", false);

  bb = base+"/saveAsText";
  pSaveAsTextCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition to root output files, also write .txt files");
  pSaveAsTextCmd->SetGuidance(guidance);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTEPCActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if(cmd == pPressureCmd) pActor->BuildMaterial(  pPressureCmd->GetNewDoubleValue(newValue)  );
  if(cmd == pEminCmd) pActor->SetEmin(  pEminCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pEmaxCmd) pActor->SetEmax(  pEmaxCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pEBinNumberCmd) pActor->SetEBinNumber(  pEBinNumberCmd->GetNewIntValue(newValue)  ) ;
  if(cmd == pELogscaleCmd) pActor->SetELogscale(  pELogscaleCmd->GetNewBoolValue(newValue)  );
  if(cmd == pENOrdersCmd) pActor->SetENOrders(  pENOrdersCmd->GetNewIntValue(newValue)  );
  if(cmd == pNormByEventCmd) pActor->SetNormByEvent(  pNormByEventCmd->GetNewBoolValue(newValue)  );
  if(cmd == pSaveAsTextCmd) pActor->SetSaveAsText(  pSaveAsTextCmd->GetNewBoolValue(newValue)  ) ;
  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------

#endif
