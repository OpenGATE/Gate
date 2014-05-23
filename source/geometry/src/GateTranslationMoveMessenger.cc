/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateTranslationMoveMessenger.hh"
#include "GateTranslationMove.hh"


#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateTranslationMoveMessenger::GateTranslationMoveMessenger(GateTranslationMove* itsTranslationMove)
  :GateObjectRepeaterMessenger(itsTranslationMove)
{ 
    G4String cmdName;

    cmdName = GetDirectoryName()+"setSpeed";
    TranslationVelocityCmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
    TranslationVelocityCmd->SetGuidance("Set the translation velocity vector dM/dt.");
    TranslationVelocityCmd->SetParameterName("dX/dt","dY/dt","dZ/dt",false);
    TranslationVelocityCmd->SetUnitCategory("Speed");

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateTranslationMoveMessenger::~GateTranslationMoveMessenger()
{
    delete TranslationVelocityCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateTranslationMoveMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command==TranslationVelocityCmd )
    { 
     G4cout << "Translation speed = " << newValue << G4endl;
     GetTranslationMove()->SetVelocity(TranslationVelocityCmd->GetNew3VectorValue(newValue));}      
  
  else 
    GateObjectRepeaterMessenger::SetNewValue(command,newValue);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
