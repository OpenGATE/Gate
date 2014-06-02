/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVolumePlacementMessenger.hh"
#include "GateVolumePlacement.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//--------------------------------------------------------------------
GateVolumePlacementMessenger::GateVolumePlacementMessenger(GateVolumePlacement* itsPlacementMove)
  :GateObjectRepeaterMessenger(itsPlacementMove)
{ 
    G4String cmdName;

    cmdName = GetDirectoryName()+"setTranslation";
    TranslationCmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
    TranslationCmd->SetGuidance("Set the translation vector.");
    TranslationCmd->SetParameterName("X0","Y0","Z0",false);
    TranslationCmd->SetUnitCategory("Length");

    cmdName = GetDirectoryName()+"setRotationAxis";
    RotationAxisCmd = new G4UIcmdWith3Vector(cmdName,this);
    RotationAxisCmd->SetGuidance("Set the rotation axis.");
    RotationAxisCmd->SetParameterName("cosAlpha","cosBeta","cosGamma",false);
    RotationAxisCmd->SetRange("cosAlpha != 0 || cosBeta != 0 || cosGamma != 0");

    cmdName = GetDirectoryName()+"setRotationAngle";
    RotationAngleCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    RotationAngleCmd->SetGuidance("Set the rotation angle.");
    RotationAngleCmd->SetParameterName("phi",false);
    RotationAngleCmd->SetUnitCategory("Angle");

    cmdName = GetDirectoryName()+"alignToX";
    AlignToXCmd = new G4UIcmdWithoutParameter(cmdName,this);
    AlignToXCmd->SetGuidance("Rotates the object by +90 deg around the Y-axis.");

    cmdName = GetDirectoryName()+"alignToY";
    AlignToYCmd = new G4UIcmdWithoutParameter(cmdName,this);
    AlignToYCmd->SetGuidance("Rotates the object by -90 deg around the Z-axis.");

    cmdName = GetDirectoryName()+"alignToZ";
    AlignToZCmd = new G4UIcmdWithoutParameter(cmdName,this);
    AlignToZCmd->SetGuidance("Resets the object orientation (rotation angle set to 0).");

    cmdName = GetDirectoryName()+"setPhiOfTranslation";
    SetPhiCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    SetPhiCmd->SetGuidance("Set the phi angle (in XY-plane) of the translation vector.");
    SetPhiCmd->SetParameterName("phi",false);
    SetPhiCmd->SetUnitCategory("Angle");

    cmdName = GetDirectoryName()+"setThetaOfTranslation";
    SetThetaCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    SetThetaCmd->SetGuidance("Set the theta angle (with regards to the Z axis) of the translation vector.");
    SetThetaCmd->SetParameterName("theta",false);
    SetThetaCmd->SetUnitCategory("Angle");

    cmdName = GetDirectoryName()+"setMagOfTranslation";
    SetMagCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    SetMagCmd->SetGuidance("Set the magnitude of the translation vector.");
    SetMagCmd->SetParameterName("mag",false);
    SetMagCmd->SetUnitCategory("Length");

}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
GateVolumePlacementMessenger::~GateVolumePlacementMessenger()
{
    delete TranslationCmd;
    delete RotationAxisCmd;
    delete RotationAngleCmd;
    delete AlignToXCmd;
    delete AlignToYCmd;
    delete AlignToZCmd;
    delete SetPhiCmd;
    delete SetThetaCmd;
    delete SetMagCmd;
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
void GateVolumePlacementMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{  
  //G4cout << " GateVolumePlacementMessenger::SetNewValue = " << newValue << G4endl;
  if( command==TranslationCmd )
    { GetVolumePlacement()->SetTranslation(TranslationCmd->GetNew3VectorValue(newValue)); /*TellGeometryToUpdate();*/}   
  
  else if( command==RotationAxisCmd )
    { GetVolumePlacement()->SetRotationAxis(RotationAxisCmd->GetNew3VectorValue(newValue)); /*TellGeometryToUpdate();*/}   
  
  else if( command==RotationAngleCmd )
  
    { GetVolumePlacement()->SetRotationAngle(RotationAngleCmd->GetNewDoubleValue(newValue)); /*TellGeometryToUpdate();*/}   
  
  else if( command==AlignToXCmd )
    { GetVolumePlacement()->AlignToX();  /*TellGeometryToUpdate();*/}   

  else if( command==AlignToYCmd )
    { GetVolumePlacement()->AlignToY();  /*TellGeometryToUpdate();*/}   

  else if( command==AlignToZCmd )
    { GetVolumePlacement()->AlignToZ(); /* TellGeometryToUpdate();*/}   

  else if( command==SetPhiCmd )
    { GetVolumePlacement()->SetPhi(SetPhiCmd->GetNewDoubleValue(newValue)); /*TellGeometryToUpdate();*/}   
  
  else if( command==SetThetaCmd )
    { GetVolumePlacement()->SetTheta(SetThetaCmd->GetNewDoubleValue(newValue)); /*TellGeometryToUpdate();*/}   
  
   else if( command==SetMagCmd )
    { GetVolumePlacement()->SetMag(SetMagCmd->GetNewDoubleValue(newValue)); /*TellGeometryToUpdate();*/}   
  
 else 
    GateObjectRepeaterMessenger::SetNewValue(command,newValue);
}
//--------------------------------------------------------------------




