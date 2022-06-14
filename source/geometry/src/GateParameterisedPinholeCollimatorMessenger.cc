/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateParameterisedPinholeCollimatorMessenger.hh"
#include "GateVVolume.hh"
#include "GateVisAttributesMessenger.hh"
#include "GateParameterisedPinholeCollimator.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithABool.hh"

GateParameterisedPinholeCollimatorMessenger::GateParameterisedPinholeCollimatorMessenger(GateParameterisedPinholeCollimator *itsInserter)
  :GateVolumeMessenger(itsInserter)
{
  name_Geometry = GetDirectoryName()+"geometry/";
  dir_Geometry = new G4UIdirectory (name_Geometry);

  dir_Geometry->SetGuidance("Control the collimator geometry.");

  G4String cmdName;

  cmdName = name_Geometry+"setDimensionX1";
  CollimatorDimensionX1Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  CollimatorDimensionX1Cmd->SetGuidance("Set x1-dimension of parameterised collimator.");
  CollimatorDimensionX1Cmd->SetParameterName("DimensionX1",false);
  CollimatorDimensionX1Cmd->SetRange("DimensionX1>0.");
  CollimatorDimensionX1Cmd->SetUnitCategory("Length");

  cmdName = name_Geometry+"setDimensionY1";
  CollimatorDimensionY1Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  CollimatorDimensionY1Cmd->SetGuidance("Set y1-dimension of parameterised collimator.");
  CollimatorDimensionY1Cmd->SetParameterName("DimensionY1",false);
  CollimatorDimensionY1Cmd->SetRange("DimensionY1>0.");
  CollimatorDimensionY1Cmd->SetUnitCategory("Length");

  cmdName = name_Geometry+"setDimensionX2";
  CollimatorDimensionX2Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  CollimatorDimensionX2Cmd->SetGuidance("Set x2-dimension of parameterised collimator.");
  CollimatorDimensionX2Cmd->SetParameterName("DimensionX2",false);
  CollimatorDimensionX2Cmd->SetRange("DimensionX2>0.");
  CollimatorDimensionX2Cmd->SetUnitCategory("Length");

  cmdName = name_Geometry+"setDimensionY2";
  CollimatorDimensionY2Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  CollimatorDimensionY2Cmd->SetGuidance("Set y2-dimension of parameterised collimator.");
  CollimatorDimensionY2Cmd->SetParameterName("DimensionY2",false);
  CollimatorDimensionY2Cmd->SetRange("DimensionY2>0.");
  CollimatorDimensionY2Cmd->SetUnitCategory("Length");



  
  cmdName = name_Geometry+"setHeight";
  CollimatorHeightCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  CollimatorHeightCmd->SetGuidance("Set height of parameterised collimator.");
  CollimatorHeightCmd->SetParameterName("Height",false);
  CollimatorHeightCmd->SetRange("Height>0.");
  CollimatorHeightCmd->SetUnitCategory("Length");

  cmdName = name_Geometry+"setRotRadius";
  CollimatorRotRadiusCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  CollimatorRotRadiusCmd->SetGuidance("Set Rotation Radius of parameterised collimator.");
  CollimatorRotRadiusCmd->SetParameterName("RotRadius",false);
  CollimatorRotRadiusCmd->SetRange("RotRadius>0.");
  CollimatorRotRadiusCmd->SetUnitCategory("Length");
  
 
  cmdName = name_Geometry+"input";
  CollimatorInpitFileCmd = new G4UIcmdWithAString(cmdName.c_str(),this);
  CollimatorInpitFileCmd->SetGuidance("Set input file with the geometry of pinhole collimator.");


  visAttributesMessenger = new GateVisAttributesMessenger(GetVolumeCreator()->GetCreator()->GetVisAttributes(),
                                                          GetVolumeCreator()->GetObjectName()+"/vis");
}

GateParameterisedPinholeCollimatorMessenger::~GateParameterisedPinholeCollimatorMessenger()
{
  delete dir_Geometry;

  delete CollimatorDimensionX1Cmd;
  delete CollimatorDimensionY1Cmd;
  delete CollimatorDimensionX2Cmd;
  delete CollimatorDimensionY2Cmd;
  delete CollimatorHeightCmd;
  delete CollimatorRotRadiusCmd;
  delete CollimatorInpitFileCmd;

  delete visAttributesMessenger;
}


void GateParameterisedPinholeCollimatorMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if( command==CollimatorDimensionX1Cmd )
    { GetCollimatorInserter()->SetCollimatorDimensionX1(CollimatorDimensionX1Cmd->GetNewDoubleValue(newValue));}

  else if( command==CollimatorDimensionY1Cmd )
    { GetCollimatorInserter()->SetCollimatorDimensionY1(CollimatorDimensionY1Cmd->GetNewDoubleValue(newValue));}

  if( command==CollimatorDimensionX2Cmd )
    { GetCollimatorInserter()->SetCollimatorDimensionX2(CollimatorDimensionX2Cmd->GetNewDoubleValue(newValue));}

  else if( command==CollimatorDimensionY2Cmd )
    { GetCollimatorInserter()->SetCollimatorDimensionY2(CollimatorDimensionY2Cmd->GetNewDoubleValue(newValue));}


   else if( command==CollimatorHeightCmd )
    { GetCollimatorInserter()->SetCollimatorHeight(CollimatorHeightCmd->GetNewDoubleValue(newValue));}
  
   else if( command==CollimatorRotRadiusCmd )
    { GetCollimatorInserter()->SetCollimatorRotRadius(CollimatorRotRadiusCmd->GetNewDoubleValue(newValue));}

   else if( command==CollimatorInpitFileCmd )
    { GetCollimatorInserter()->SetCollimatorInputFile(newValue);}
  
   else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
