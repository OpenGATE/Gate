/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT
#include "GateARFSDMessenger.hh"
#include "GateARFSD.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
/* Constructor */
/* 'itsName' is the base-name of the directory
   The flag 'flagCreateDirectory' tells whether it should create a new UI directory
   (set this flag to false if the directory is already created by another messenger) */

GateARFSDMessenger::GateARFSDMessenger(GateARFSD* ARFSD) :
  GateMessenger(ARFSD->GetName()), mArfSD(ARFSD)
{
  G4String cmdName;
  cmdName = GetDirectoryName() + "setProjectionPlane";
  mSetDepth = new G4UIcmdWithADoubleAndUnit(cmdName, this);
  mSetDepth->SetGuidance("sets the YZ projection plane relative to the ARF device center");
  cmdName = GetDirectoryName() + "setEnergyDepositionThreshHold";
  mSetEThreshHoldcmd = new G4UIcmdWithADoubleAndUnit(cmdName, this);
}

GateARFSDMessenger::~GateARFSDMessenger()
{
  delete mSetDepth;
  delete mSetEThreshHoldcmd;
}

/* UI command interpreter method */
void GateARFSDMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == mSetDepth)
    {
      mArfSD->SetDepth(mSetDepth->GetNewDoubleValue(newValue));
      return;
    }
  if (command == mSetEThreshHoldcmd)
    {
      mArfSD->setEnergyDepositionThreshold(mSetEThreshHoldcmd->GetNewDoubleValue(newValue));
    }
}
#endif
