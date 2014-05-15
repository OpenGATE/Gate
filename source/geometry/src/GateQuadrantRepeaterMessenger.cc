/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateQuadrantRepeaterMessenger.hh"
#include "GateQuadrantRepeater.hh"


#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWith3Vector.hh"


//---------------------------------------------------------------------------------------------------------
GateQuadrantRepeaterMessenger::GateQuadrantRepeaterMessenger(GateQuadrantRepeater* itsQuadrantRepeater)
  :GateObjectRepeaterMessenger(itsQuadrantRepeater)
{ 
    G4String cmdName;

    cmdName = GetDirectoryName()+"setLineNumber";
    SetLineNumberCmd = new G4UIcmdWithAnInteger(cmdName,this);
    SetLineNumberCmd->SetGuidance("Set the number of repetition lines.");
    SetLineNumberCmd->SetParameterName("N",false);
    SetLineNumberCmd->SetRange("N >= 1");

    cmdName = GetDirectoryName()+"setOrientation";
    SetOrientationCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    SetOrientationCmd->SetGuidance("Set the orientation of the quadrant (the direction of line repetition).");
    SetOrientationCmd->SetParameterName("angle",false);
    SetOrientationCmd->SetUnitCategory("Angle");

    cmdName = GetDirectoryName()+"setCopySpacing";
    SetCopySpacingCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    SetCopySpacingCmd->SetGuidance("Set the distance between adjacent copies.");
    SetCopySpacingCmd->SetParameterName("dist",false);
    SetCopySpacingCmd->SetUnitCategory("Length");

    cmdName = GetDirectoryName()+"setMaxRange";
    SetMaxRangeCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    SetMaxRangeCmd->SetGuidance("Set the maximum range of the repeater.");
    SetMaxRangeCmd->SetGuidance("The repeater range is the maximum distance between a copy and the original object.");
    SetMaxRangeCmd->SetGuidance("Use this command to remove corner-copies that would fall outside your phantom");
    SetMaxRangeCmd->SetParameterName("range",false);
    SetMaxRangeCmd->SetUnitCategory("Length");
    SetMaxRangeCmd->SetRange("range >= 0");

}
//---------------------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------------------
GateQuadrantRepeaterMessenger::~GateQuadrantRepeaterMessenger()
{
    delete SetOrientationCmd;
    delete SetLineNumberCmd;
    delete SetCopySpacingCmd;
    delete SetMaxRangeCmd;
}
//---------------------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------------------
void GateQuadrantRepeaterMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command==SetOrientationCmd )
    { GetQuadrantRepeater()->SetOrientation(SetOrientationCmd->GetNewDoubleValue(newValue));}   
  else if( command==SetCopySpacingCmd )
    { GetQuadrantRepeater()->SetCopySpacing(SetCopySpacingCmd->GetNewDoubleValue(newValue));}   
  else if( command==SetLineNumberCmd )
    { GetQuadrantRepeater()->SetLineNumber(SetLineNumberCmd->GetNewIntValue(newValue));}   
  else if( command==SetMaxRangeCmd )
    { GetQuadrantRepeater()->SetMaxRange(SetMaxRangeCmd->GetNewDoubleValue(newValue));}   
  else 
    GateObjectRepeaterMessenger::SetNewValue(command,newValue);
}
//---------------------------------------------------------------------------------------------------------
