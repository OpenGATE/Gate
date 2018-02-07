/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
 ----------------------*/
#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateARFDataToRootMessenger.hh"
#include "GateARFDataToRoot.hh"
#include "GateOutputMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

GateARFDataToRootMessenger::GateARFDataToRootMessenger(GateARFDataToRoot* GateARFDataToRoot) :
    GateOutputModuleMessenger(GateARFDataToRoot), mGateArfDataToRoot(GateARFDataToRoot)
  {
  G4String cmdName;
  cmdName = GetDirectoryName() + "setFileName";
  mSetArfDataFileCmd = new G4UIcmdWithAString(cmdName, this);
  mSetArfDataFileCmd->SetGuidance("sets the ARF Data Root File Name");
  cmdName = GetDirectoryName() + "setProjectionPlane";
  mSetDepth = new G4UIcmdWithADoubleAndUnit(cmdName, this);
  mSetDepth->SetGuidance("sets the YZ projection plane relative to the ARF device center");
  cmdName = GetDirectoryName() + "applyToDRFData";
  mSmoothDrfCmd = new G4UIcmdWithAString(cmdName, this);

  }

GateARFDataToRootMessenger::~GateARFDataToRootMessenger()
  {

  delete mSetArfDataFileCmd;
  delete mSetDepth;
  delete mSmoothDrfCmd;
  }

void GateARFDataToRootMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
  {

  if (command == mSmoothDrfCmd)
    {
    if (newValue == "smoothness")
      {
      mGateArfDataToRoot->setDRFDataprojectionmode(0);
      return;
      }
    if (newValue == "line-projection")
      {
      mGateArfDataToRoot->setDRFDataprojectionmode(1);
      return;
      }
    if (newValue == "orthogonal-projection")
      {
      mGateArfDataToRoot->setDRFDataprojectionmode(2);
      return;
      }
    G4cout << " GateARFSimuSDMessenger::SetNewValue ::: UNKNOWN parameter "
           << newValue
           << ". Ignored DRF Data projection Mode. Set To Default : smoothness \n";
    return;
    }

  if (command == mSetDepth)
    {
    mGateArfDataToRoot->SetProjectionPlane(mSetDepth->GetNewDoubleValue(newValue));
    return;
    }

  if (command == mSetArfDataFileCmd)
    {
    mGateArfDataToRoot->SetARFDataRootFileName(newValue);
    return;
    }

  GateOutputModuleMessenger::SetNewValue(command, newValue);
  }

#endif
