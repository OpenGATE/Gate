/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
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

GateARFTableMgrMessenger::GateARFTableMgrMessenger(G4String tableName, GateARFTableMgr* ARFTableMgr)
  {
  mArfTableMgr = ARFTableMgr;
  G4String dirName = "/gate/" + tableName + "/ARFTables/";
  mGateARFTableDir = new G4UIdirectory(dirName);
  mGateARFTableDir->SetGuidance("GATE ARF Tables manager control.");
  G4String cmdName = dirName + "ComputeTablesFromEnergyWindows";
  mCptTableEWCmd = new G4UIcmdWithAString(cmdName, this);
  mCptTableEWCmd->SetGuidance("Compute the ARF Tables from the energy windows data file");
  mCptTableEWCmd->SetGuidance("the text file containing the energy windows root files to be used to generate the ARF tables");
  cmdName = dirName + "list";
  mListARFTableCmd = new G4UIcmdWithoutParameter(cmdName, this);
  mListARFTableCmd->SetGuidance("List of the defined ARF tables");
  cmdName = dirName + "verbose";
  mVerboseCmd = new G4UIcmdWithAnInteger(cmdName, this);
  mVerboseCmd->SetGuidance("Set verbose level");
  mVerboseCmd->SetGuidance("1. Integer verbose level");
  mVerboseCmd->SetParameterName("verbose", false);
  mVerboseCmd->SetRange("verbose>=0");

  cmdName = dirName + "output/setNBins";
  mSetNBinsCmd = new G4UIcmdWithAnInteger(cmdName, this);

  cmdName = dirName + "setEnergyDepositionThreshHold";
  mSetEThreshHoldcmd = new G4UIcmdWithADoubleAndUnit(cmdName, this);

  cmdName = dirName + "setEnergyDepositionUpHold";
  mSetEUpHoldcmd = new G4UIcmdWithADoubleAndUnit(cmdName, this);

  cmdName = dirName + "setEnergyResolution";
  mSetEResocmd = new G4UIcmdWithADouble(cmdName, this);

  cmdName = dirName + "setEnergyOfReference";
  mSetERefcmd = new G4UIcmdWithADoubleAndUnit(cmdName, this);

  cmdName = dirName + "setDistanceFromSourceToDetector";
  mSetDistancecmd = new G4UIcmdWithADoubleAndUnit(cmdName, this);

  cmdName = dirName + "saveARFTablesToBinaryFile";
  mSaveToBinaryFileCmd = new G4UIcmdWithAString(cmdName, this);

  cmdName = dirName + "loadARFTablesFromBinaryFile";
  mLoadFromBinaryFileCmd = new G4UIcmdWithAString(cmdName, this);

  }

GateARFTableMgrMessenger::~GateARFTableMgrMessenger()
  {
  delete mListARFTableCmd;
  delete mVerboseCmd;
  delete mGateARFTableDir;
  delete mSetEResocmd;
  delete mSetERefcmd;
  delete mSetNBinsCmd;
  delete mCptTableEWCmd;
  delete mSetEThreshHoldcmd;
  delete mSetEUpHoldcmd;
  delete mSaveToBinaryFileCmd;
  delete mLoadFromBinaryFileCmd;
  delete mSetDistancecmd;
  }

void GateARFTableMgrMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
  {

  if (command == mSetDistancecmd)
    {
    mArfTableMgr->SetDistanceFromSourceToDetector(mSetDistancecmd->GetNewDoubleValue(newValue));
    return;
    }
  if (command == mSetNBinsCmd)
    {
    mArfTableMgr->SetNBins(mSetNBinsCmd->GetNewIntValue(newValue));
    return;
    }
  if (command == mLoadFromBinaryFileCmd)
    {
    mArfTableMgr->LoadARFFromBinaryFile(newValue);
    }

  if (command == mSaveToBinaryFileCmd)
    {
    mArfTableMgr->SetBinaryFile(newValue);
    mArfTableMgr->SaveARFToBinaryFile();
    }

  if (command == mSetERefcmd)
    {
    mArfTableMgr->SetERef(mSetERefcmd->GetNewDoubleValue(newValue));
    return;
    }

  if (command == mSetEResocmd)
    {
    mArfTableMgr->SetEReso(mSetEResocmd->GetNewDoubleValue(newValue));
    return;
    }

  if (command == mSetEThreshHoldcmd)
    {
    mArfTableMgr->SetEThreshHold(mSetEThreshHoldcmd->GetNewDoubleValue(newValue));
    return;
    }
  if (command == mSetEUpHoldcmd)
    {
    mArfTableMgr->SetEUpHold(mSetEUpHoldcmd->GetNewDoubleValue(newValue));
    return;
    }

  if (command == mVerboseCmd)
    {
    mArfTableMgr->SetVerboseLevel(mVerboseCmd->GetNewIntValue(newValue));
    return;
    }

  if (command == mListARFTableCmd)
    {
    mArfTableMgr->ListTables();
    return;
    }

  if (command == mCptTableEWCmd)
    {
    mArfTableMgr->ComputeARFTablesFromEW(newValue);
    return;
    }

  }

#endif
