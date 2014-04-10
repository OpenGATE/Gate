/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateParameterisedCollimatorMessenger.hh"
#include "GateVVolume.hh"
#include "GateVisAttributesMessenger.hh"
#include "GateParameterisedCollimator.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

GateParameterisedCollimatorMessenger::GateParameterisedCollimatorMessenger(GateParameterisedCollimator *itsInserter)
  :GateVolumeMessenger(itsInserter)
{
  name_Geometry = GetDirectoryName()+"geometry/";
  dir_Geometry = new G4UIdirectory (name_Geometry);

  dir_Geometry->SetGuidance("Control the collimator geometry.");

  G4String cmdName;

  cmdName = name_Geometry+"setDimensionX";
  CollimatorDimensionXCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  CollimatorDimensionXCmd->SetGuidance("Set x-dimension of parameterised collimator.");
  CollimatorDimensionXCmd->SetParameterName("DimensionX",false);
  CollimatorDimensionXCmd->SetRange("DimensionX>0.");
  CollimatorDimensionXCmd->SetUnitCategory("Length");

  cmdName = name_Geometry+"setDimensionY";
  CollimatorDimensionYCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  CollimatorDimensionYCmd->SetGuidance("Set y-dimension of parameterised collimator.");
  CollimatorDimensionYCmd->SetParameterName("DimensionY",false);
  CollimatorDimensionYCmd->SetRange("DimensionY>0.");
  CollimatorDimensionYCmd->SetUnitCategory("Length");

  cmdName = name_Geometry+"setFocalDistanceX";
  CollimatorFocalDistanceXCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  CollimatorFocalDistanceXCmd->SetGuidance("Set focal distance of parameterised collimator.");
  CollimatorFocalDistanceXCmd->SetParameterName("FocalDistanceX",false);
  //  CollimatorFocalDistanceXCmd->SetRange("FocalDistanceX>0.");
  CollimatorFocalDistanceXCmd->SetUnitCategory("Length");

  cmdName = name_Geometry+"setFocalDistanceY";
  CollimatorFocalDistanceYCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  CollimatorFocalDistanceYCmd->SetGuidance("Set focal distance of parameterised collimator.");
  CollimatorFocalDistanceYCmd->SetParameterName("FocalDistanceY",false);
  //  CollimatorFocalDistanceYCmd->SetRange("FocalDistanceY>0.");
  CollimatorFocalDistanceYCmd->SetUnitCategory("Length");

  cmdName = name_Geometry+"setHeight";
  CollimatorHeightCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  CollimatorHeightCmd->SetGuidance("Set height of parameterised collimator.");
  CollimatorHeightCmd->SetParameterName("Height",false);
  CollimatorHeightCmd->SetRange("Height>0.");
  CollimatorHeightCmd->SetUnitCategory("Length");

  cmdName = name_Geometry+"setSeptalThickness";
  CollimatorSeptalThicknessCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  CollimatorSeptalThicknessCmd->SetGuidance("Set septal thickness of parameterised collimator.");
  CollimatorSeptalThicknessCmd->SetParameterName("SeptalThickness",false);
  CollimatorSeptalThicknessCmd->SetRange("SeptalThickness>0.");
  CollimatorSeptalThicknessCmd->SetUnitCategory("Length");

  cmdName = name_Geometry+"setInnerRadius";
  CollimatorInnerRadiusCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  CollimatorInnerRadiusCmd->SetGuidance("Set inner radius of parameterised collimator.");
  CollimatorInnerRadiusCmd->SetParameterName("InnerRadius",false);
  CollimatorInnerRadiusCmd->SetRange("InnerRadius>0.");
  CollimatorInnerRadiusCmd->SetUnitCategory("Length");

  visAttributesMessenger = new GateVisAttributesMessenger(GetVolumeCreator()->GetCreator()->GetVisAttributes(),
                                                          GetVolumeCreator()->GetObjectName()+"/vis");
}

GateParameterisedCollimatorMessenger::~GateParameterisedCollimatorMessenger()
{
  delete dir_Geometry;

  delete CollimatorDimensionXCmd;
  delete CollimatorDimensionYCmd;
  delete CollimatorFocalDistanceXCmd;
  delete CollimatorFocalDistanceYCmd;
  delete CollimatorHeightCmd;
  delete CollimatorSeptalThicknessCmd;
  delete CollimatorInnerRadiusCmd;

  delete visAttributesMessenger;
}


void GateParameterisedCollimatorMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if( command==CollimatorDimensionXCmd )
    { GetCollimatorInserter()->SetCollimatorDimensionX(CollimatorDimensionXCmd->GetNewDoubleValue(newValue));}

  else if( command==CollimatorDimensionYCmd )
    { GetCollimatorInserter()->SetCollimatorDimensionY(CollimatorDimensionYCmd->GetNewDoubleValue(newValue));}

  else if( command==CollimatorFocalDistanceXCmd )
    { GetCollimatorInserter()->SetCollimatorFocalDistanceX(CollimatorFocalDistanceXCmd->GetNewDoubleValue(newValue));}

  else if( command==CollimatorFocalDistanceYCmd )
    { GetCollimatorInserter()->SetCollimatorFocalDistanceY(CollimatorFocalDistanceYCmd->GetNewDoubleValue(newValue));}

  else if( command==CollimatorHeightCmd )
    { GetCollimatorInserter()->SetCollimatorHeight(CollimatorHeightCmd->GetNewDoubleValue(newValue));}

  else if( command==CollimatorSeptalThicknessCmd )
    { GetCollimatorInserter()->SetCollimatorSeptalThickness(CollimatorSeptalThicknessCmd->GetNewDoubleValue(newValue));}

  else if( command==CollimatorInnerRadiusCmd )
    { GetCollimatorInserter()->SetCollimatorInnerRadius(CollimatorInnerRadiusCmd->GetNewDoubleValue(newValue));}

  else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
