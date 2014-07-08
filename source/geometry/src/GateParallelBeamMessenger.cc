/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateParallelBeamMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

#include "GateParallelBeam.hh"

//-------------------------------------------------------------------------------------------------
GateParallelBeamMessenger::GateParallelBeamMessenger(GateParallelBeam *itsInserter)
  :GateMessenger(itsInserter->GetObjectName()),
   m_inserter(itsInserter)
{
  GetDirectory()->SetGuidance("Control the parallel beam collimator geometry and material.");

  G4String cmdName;

  cmdName = G4String("/gate/") + itsInserter->GetObjectName() + "/setMaterialName";
  ParallelBeamMaterialCmd = new G4UIcmdWithAString(cmdName,this);
  ParallelBeamMaterialCmd->SetGuidance("Selects the material for the parallel beam collimator.");
  ParallelBeamMaterialCmd->SetParameterName("material",false);

  cmdName = GetDirectoryName()+"geometry/setDimensionX";
  ParallelBeamDimensionXCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  ParallelBeamDimensionXCmd->SetGuidance("Set x-dimension of parallel beam collimator.");
  ParallelBeamDimensionXCmd->SetParameterName("DimensionX",false);
  ParallelBeamDimensionXCmd->SetRange("DimensionX>0.");
  ParallelBeamDimensionXCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"geometry/setDimensionY";
  ParallelBeamDimensionYCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  ParallelBeamDimensionYCmd->SetGuidance("Set y-dimension of parallel beam collimator.");
  ParallelBeamDimensionYCmd->SetParameterName("DimensionY",false);
  ParallelBeamDimensionYCmd->SetRange("DimensionY>0.");
  ParallelBeamDimensionYCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"geometry/setHeight";
  ParallelBeamHeightCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  ParallelBeamHeightCmd->SetGuidance("Set height of parallel beam collimator.");
  ParallelBeamHeightCmd->SetParameterName("Height",false);
  ParallelBeamHeightCmd->SetRange("Height>0.");
  ParallelBeamHeightCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"geometry/setSeptalThickness";
  ParallelBeamSeptalThicknessCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  ParallelBeamSeptalThicknessCmd->SetGuidance("Set septal thickness of parallel beam collimator.");
  ParallelBeamSeptalThicknessCmd->SetParameterName("SeptalThickness",false);
  ParallelBeamSeptalThicknessCmd->SetRange("SeptalThickness>0.");
  ParallelBeamSeptalThicknessCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"geometry/setInnerRadius";
  ParallelBeamInnerRadiusCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  ParallelBeamInnerRadiusCmd->SetGuidance("Set inner radius of parallel beam collimator.");
  ParallelBeamInnerRadiusCmd->SetParameterName("InnerRadius",false);
  ParallelBeamInnerRadiusCmd->SetRange("InnerRadius>0.");
  ParallelBeamInnerRadiusCmd->SetUnitCategory("Length");

}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
GateParallelBeamMessenger::~GateParallelBeamMessenger()
{
  delete ParallelBeamDimensionXCmd;
  delete ParallelBeamDimensionYCmd;
  delete ParallelBeamHeightCmd;
  delete ParallelBeamSeptalThicknessCmd;
  delete ParallelBeamInnerRadiusCmd;
  delete ParallelBeamMaterialCmd;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateParallelBeamMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if( command==ParallelBeamDimensionXCmd )
    { GetParallelBeamInserter()->SetParallelBeamDimensionX(ParallelBeamDimensionXCmd->GetNewDoubleValue(newValue));}

  else if( command==ParallelBeamDimensionYCmd )
    { GetParallelBeamInserter()->SetParallelBeamDimensionY(ParallelBeamDimensionYCmd->GetNewDoubleValue(newValue));}

  else if( command==ParallelBeamHeightCmd )
    { GetParallelBeamInserter()->SetParallelBeamHeight(ParallelBeamHeightCmd->GetNewDoubleValue(newValue));}

  else if( command==ParallelBeamSeptalThicknessCmd )
    { GetParallelBeamInserter()->SetParallelBeamSeptalThickness(ParallelBeamSeptalThicknessCmd->GetNewDoubleValue(newValue));}

  else if( command==ParallelBeamInnerRadiusCmd )
    { GetParallelBeamInserter()->SetParallelBeamInnerRadius(ParallelBeamInnerRadiusCmd->GetNewDoubleValue(newValue));}

  else if( command==ParallelBeamMaterialCmd )
    { GetParallelBeamInserter()->SetParallelBeamMaterial(newValue);}


  else
    GateMessenger::SetNewValue(command,newValue);

}
//-------------------------------------------------------------------------------------------------
