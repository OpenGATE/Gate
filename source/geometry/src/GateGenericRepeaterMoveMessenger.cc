/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateGenericRepeaterMoveMessenger.hh"
#include "GateGenericRepeater.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"

//--------------------------------------------------------------------------------------------
GateGenericRepeaterMoveMessenger::GateGenericRepeaterMoveMessenger(GateGenericRepeaterMove * itsRepeater)
  :GateObjectRepeaterMessenger(itsRepeater) { 
  G4String cmdName;
  
  cmdName = GetDirectoryName()+"setPlacementsFilename";
  mFilenameCmd = new G4UIcmdWithAString(cmdName,this);
  mFilenameCmd->SetGuidance("Set the filename containing the list of repeated placement for each time stamp."); 

  cmdName = GetDirectoryName()+"useRelativeTranslation";
  mRelativeTransCmd = new G4UIcmdWithABool(cmdName,this);
  mRelativeTransCmd->SetGuidance("If true, translation are relative to the initial translation."); 
}
//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
GateGenericRepeaterMoveMessenger::~GateGenericRepeaterMoveMessenger() {
  delete mFilenameCmd;
  delete mRelativeTransCmd;
}
//--------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------
void GateGenericRepeaterMoveMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if (command==mFilenameCmd) GetGenericRepeaterMove()->SetFilename(newValue);
  else if (command == mRelativeTransCmd) GetGenericRepeaterMove()->EnableRelativeTranslation(mRelativeTransCmd->GetNewBoolValue(newValue));
  else GateObjectRepeaterMessenger::SetNewValue(command,newValue);
}
//--------------------------------------------------------------------------------------------
