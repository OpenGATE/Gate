/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateToProjectionSetMessenger.hh"
#include "GateToProjectionSet.hh"
#include "GateOutputMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"





GateToProjectionSetMessenger::GateToProjectionSetMessenger(GateToProjectionSet* gateToProjectionSet)
  : GateOutputModuleMessenger(gateToProjectionSet)
  , m_gateToProjectionSet(gateToProjectionSet)
{
  G4String cmdName;

  cmdName = GetDirectoryName()+"setFileName";
  SetFileNameCmd = new G4UIcmdWithAString(cmdName,this);
  SetFileNameCmd->SetGuidance("Set the name of the output Interfile file containing the projections");
  SetFileNameCmd->SetParameterName("Name",false);

  cmdName = GetDirectoryName()+"projectionPlane";
  projectionPlaneCmd = new G4UIcmdWithAString(cmdName.c_str(),this);
  projectionPlaneCmd->SetGuidance("Defines which projection plane to use");
  projectionPlaneCmd->SetParameterName("choice",false);
  projectionPlaneCmd->SetCandidates("XY YZ ZX");

  cmdName = GetDirectoryName()+"pixelSizeX";
  PixelSizeXCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  PixelSizeXCmd->SetGuidance("Set the pixel size along X.");
  PixelSizeXCmd->SetParameterName("Size",false);
  PixelSizeXCmd->SetRange("Size>0.");
  PixelSizeXCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"pixelSizeY";
  PixelSizeYCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  PixelSizeYCmd->SetGuidance("Set the pixel size along Y.");
  PixelSizeYCmd->SetParameterName("Size",false);
  PixelSizeYCmd->SetRange("Size>0.");
  PixelSizeYCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"pixelNumberX";
  PixelNumberXCmd = new G4UIcmdWithAnInteger(cmdName.c_str(),this);
  PixelNumberXCmd->SetGuidance("Set the number of pixels along X.");
  PixelNumberXCmd->SetParameterName("Number",false);
  PixelNumberXCmd->SetRange("Number>0");

  cmdName = GetDirectoryName()+"pixelNumberY";
  PixelNumberYCmd = new G4UIcmdWithAnInteger(cmdName.c_str(),this);
  PixelNumberYCmd->SetGuidance("Set the number of pixels along Y.");
  PixelNumberYCmd->SetParameterName("Number",false);
  PixelNumberYCmd->SetRange("Number>0");

  cmdName = GetDirectoryName()+"setInputDataName";
  SetInputDataCmd = new G4UIcmdWithAString(cmdName,this);
  SetInputDataCmd->SetGuidance("Set the name of the input data to store into the sinogram");
  SetInputDataCmd->SetParameterName("Name",false);

  cmdName = GetDirectoryName()+"addInputDataName";
  AddInputDataCmd = new G4UIcmdWithAString(cmdName,this);
  AddInputDataCmd->SetGuidance("Add the name of the input data to store into the sinogram");
  AddInputDataCmd->SetParameterName("Name",false);

}





GateToProjectionSetMessenger::~GateToProjectionSetMessenger()
{
  delete SetFileNameCmd;
  delete PixelSizeXCmd;
  delete PixelSizeYCmd;
  delete PixelNumberXCmd;
  delete PixelNumberYCmd;
  delete projectionPlaneCmd;
  delete SetInputDataCmd;
  delete AddInputDataCmd;
}




void GateToProjectionSetMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if( command==projectionPlaneCmd )
    { m_gateToProjectionSet->SetProjectionPlane(newValue); }

  else if (command == SetFileNameCmd)
    { m_gateToProjectionSet->SetOutputFileName(newValue); }

  else if( command==PixelSizeXCmd )
    { m_gateToProjectionSet->SetPixelSizeX(PixelSizeXCmd->GetNewDoubleValue(newValue)); }

  else if( command==PixelSizeYCmd )
    { m_gateToProjectionSet->SetPixelSizeY(PixelSizeYCmd->GetNewDoubleValue(newValue)); }

  else if( command==PixelNumberXCmd )
    { m_gateToProjectionSet->SetPixelNbX(PixelNumberXCmd->GetNewIntValue(newValue)); }

  else if( command==PixelNumberYCmd )
    { m_gateToProjectionSet->SetPixelNbY(PixelNumberYCmd->GetNewIntValue(newValue)); }

  else if (command == SetInputDataCmd)
    { m_gateToProjectionSet->SetInputDataName(newValue); }

  else if (command == AddInputDataCmd)
    { m_gateToProjectionSet->AddInputDataName(newValue); }

  /*
   * Commands of the mother overloaded to have impact on the GateToInterfile class too
   */
  else if( command == GetVerboseCmd() )
    { m_gateToProjectionSet->SetVerboseToProjectionSetAndInterfile(GetVerboseCmd()->GetNewIntValue(newValue)); }

  else if( command == GetDescribeCmd() )
    { m_gateToProjectionSet->SendDescribeToProjectionSetAndInterfile(); }

  else if ( command == GetEnableCmd() )
    { m_gateToProjectionSet->SetEnableToProjectionSetAndInterfile(); }

  else if ( command == GetDisableCmd() )
    { m_gateToProjectionSet->SetDisableToProjectionSetAndInterfile(); }

  // Other commands are passed to the mother (but normally nothing !)
  else
    { GateOutputModuleMessenger::SetNewValue(command,newValue); }
}
