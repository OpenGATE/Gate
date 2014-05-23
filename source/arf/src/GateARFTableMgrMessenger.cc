/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateARFTableMgrMessenger.hh"
#include "GateClock.hh"
#include "GateARFTableMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateARFTableMgrMessenger::GateARFTableMgrMessenger(G4String aName, GateARFTableMgr* ARFTableMgr)
{
  m_ARFTableMgr = ARFTableMgr;

  G4String dirName = "/gate/"+aName+"/ARFTables/";

  GateARFTableDir = new G4UIdirectory(dirName);
  GateARFTableDir->SetGuidance("GATE ARF Tables manager control.");

  G4String cmdName = dirName+"ComputeTablesFromEnergyWindows";
  cptTableEWCmd = new G4UIcmdWithAString(cmdName,this);
  cptTableEWCmd->SetGuidance("Compute the ARF Tables from the energy windows data file");
  cptTableEWCmd->SetGuidance("the text file containing the energy windows root files to be used to generate the ARF tables");


  cmdName = dirName+"list";
  ListARFTableCmd = new G4UIcmdWithoutParameter(cmdName,this);
  ListARFTableCmd->SetGuidance("List of the defined ARF tables");



  cmdName = dirName+"verbose";
  VerboseCmd = new G4UIcmdWithAnInteger(cmdName,this);
  VerboseCmd->SetGuidance("Set verbose level");
  VerboseCmd->SetGuidance("1. Integer verbose level");
  VerboseCmd->SetParameterName("verbose",false);
  VerboseCmd->SetRange("verbose>=0");

  cmdName = dirName+"output/setNBins";
  SetNBinsCmd = new G4UIcmdWithAnInteger(cmdName,this);


  cmdName = dirName+"setEnergyDepositionThreshHold";
  setEThreshHoldcmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);


  cmdName = dirName+"setEnergyDepositionUpHold";
  setEUpHoldcmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);



  cmdName = dirName+"setEnergyResolution";
  setEResocmd = new G4UIcmdWithADouble(cmdName,this);


  cmdName = dirName+"setEnergyOfReference";
  setERefcmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);


  cmdName = dirName+"setDistanceFromSourceToDetector";
  setDistancecmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);


  cmdName = dirName+"saveARFTablesToBinaryFile";
  SaveToBinaryFileCmd = new G4UIcmdWithAString(cmdName,this);


  cmdName = dirName+"loadARFTablesFromBinaryFile";
  LoadFromBinaryFileCmd = new G4UIcmdWithAString(cmdName,this);




}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateARFTableMgrMessenger::~GateARFTableMgrMessenger()
{
  delete ListARFTableCmd;
  delete VerboseCmd;
  delete GateARFTableDir;
  delete setEResocmd;
  delete setERefcmd;
  delete SetNBinsCmd;
  delete cptTableEWCmd;
  delete setEThreshHoldcmd ;
  delete   setEUpHoldcmd ;
  delete SaveToBinaryFileCmd;
  delete LoadFromBinaryFileCmd;
  delete setDistancecmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateARFTableMgrMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{


  if ( command == setDistancecmd ) { m_ARFTableMgr->SetDistanceFromSourceToDetector( setDistancecmd->GetNewDoubleValue(newValue) );
    return; }
  if( command == SetNBinsCmd)
    {
      m_ARFTableMgr->SetNBins(SetNBinsCmd->GetNewIntValue(newValue));
      return;
    }
  if ( command == LoadFromBinaryFileCmd ) { m_ARFTableMgr->LoadARFFromBinaryFile(newValue);}

  if ( command == SaveToBinaryFileCmd ) {
    m_ARFTableMgr->SetBinaryFile(newValue);
    m_ARFTableMgr->SaveARFToBinaryFile();
  }

  if ( command == setERefcmd ) { m_ARFTableMgr->SetERef( setERefcmd->GetNewDoubleValue(newValue) );
    return; }

  if ( command == setEResocmd ) { m_ARFTableMgr->SetEReso( setEResocmd->GetNewDoubleValue(newValue) );
    return; }

  if ( command == setEThreshHoldcmd ){ m_ARFTableMgr->SetEThreshHold( setEThreshHoldcmd->GetNewDoubleValue(newValue) ) ;return;}
  if ( command ==   setEUpHoldcmd  ){ m_ARFTableMgr->SetEUpHold( setEUpHoldcmd->GetNewDoubleValue(newValue) ) ;return;}

  if( command == VerboseCmd ) {
    m_ARFTableMgr->SetVerboseLevel(VerboseCmd->GetNewIntValue(newValue));
    return;
  }

  if( command == ListARFTableCmd ) {
    m_ARFTableMgr->ListTables();
    return;
  }

  if( command == cptTableEWCmd ) {
    m_ARFTableMgr->ComputeARFTablesFromEW(newValue);return;
  }


}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#endif
