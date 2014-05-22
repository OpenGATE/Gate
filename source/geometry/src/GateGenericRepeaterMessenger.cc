/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateGenericRepeaterMessenger.hh"
#include "GateGenericRepeater.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"

//--------------------------------------------------------------------------------------------
GateGenericRepeaterMessenger::GateGenericRepeaterMessenger(GateGenericRepeater* itsRepeater)
  :GateObjectRepeaterMessenger(itsRepeater) { 
  G4String cmdName;  
  cmdName = GetDirectoryName()+"setPlacementsFilename";
  mFileCmd = new G4UIcmdWithAString(cmdName,this);
  mFileCmd->SetGuidance("Set a filename with a list of 3D translations and rotations.");

  cmdName = GetDirectoryName()+"useRelativeTranslation";
  mRelativeTransCmd = new G4UIcmdWithABool(cmdName,this);
  mRelativeTransCmd->SetGuidance("If true, translation are relative to the initial translation.");
}
//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
GateGenericRepeaterMessenger::~GateGenericRepeaterMessenger() {
  delete mFileCmd;
  delete mRelativeTransCmd;
}
//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
void GateGenericRepeaterMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if (command == mFileCmd) GetGenericRepeater()->SetPlacementsFilename(newValue);
  else if (command == mRelativeTransCmd) GetGenericRepeater()->EnableRelativeTranslation(mRelativeTransCmd->GetNewBoolValue(newValue));
  else GateObjectRepeaterMessenger::SetNewValue(command,newValue);
}
//--------------------------------------------------------------------------------------------
