/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateToSinoAccelMessenger.hh"
#include "GateToSinoAccel.hh"
#include "GateOutputMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"





GateToSinoAccelMessenger::GateToSinoAccelMessenger(GateToSinoAccel* gateToSinoAccel)
  : GateOutputModuleMessenger(gateToSinoAccel)
  , m_gateToSinoAccel(gateToSinoAccel)
{
  G4String cmdName;

  cmdName = GetDirectoryName()+"setFileName";
  SetFileNameCmd = new G4UIcmdWithAString(cmdName,this);
  SetFileNameCmd->SetGuidance("Set the name of the output raw sinogram files");
  SetFileNameCmd->SetParameterName("Name",false);

  cmdName = GetDirectoryName()+"TruesOnly";
  TruesOnlyCmd = new G4UIcmdWithABool(cmdName,this);
  TruesOnlyCmd->SetGuidance("Record only true coincidences");
  TruesOnlyCmd->SetParameterName("flag",true);
  TruesOnlyCmd->SetDefaultValue(true);

  cmdName = GetDirectoryName()+"RadialBins";
  SetRadialElemNbCmd = new G4UIcmdWithAnInteger(cmdName,this);
  SetRadialElemNbCmd->SetGuidance("Set the number of radial sinogram bins");
  SetRadialElemNbCmd->SetParameterName("Number",false);
  SetRadialElemNbCmd->SetRange("Number>0");

  cmdName = GetDirectoryName()+"RawOutputEnable";
  RawOutputCmd = new G4UIcmdWithABool(cmdName,this);
  RawOutputCmd->SetGuidance("Enables 2D sinograms output in a raw file");
  RawOutputCmd->SetParameterName("flag",true);
  RawOutputCmd->SetDefaultValue(true);

  cmdName = GetDirectoryName()+"setInputDataName";
  SetInputDataCmd = new G4UIcmdWithAString(cmdName,this);
  SetInputDataCmd->SetGuidance("Set the name of the input data to store into the sinogram");
  SetInputDataCmd->SetParameterName("Name",false);

  cmdName = GetDirectoryName()+"setTangCrystalBlurring";
  SetTangCrystalResolCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  SetTangCrystalResolCmd->SetGuidance("Set the crystal location blurring FWHM in the tangential direction");
  SetTangCrystalResolCmd->SetParameterName("Number",false);
  SetTangCrystalResolCmd->SetRange("Number>=0.");
  SetTangCrystalResolCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"setAxialCrystalBlurring";
  SetAxialCrystalResolCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  SetAxialCrystalResolCmd->SetGuidance("Set the crystal location blurring FWHM in the axial direction");
  SetAxialCrystalResolCmd->SetParameterName("Number",false);
  SetAxialCrystalResolCmd->SetRange("Number>=0.");
  SetAxialCrystalResolCmd->SetUnitCategory("Length");

}
GateToSinoAccelMessenger::~GateToSinoAccelMessenger()
{
  delete SetFileNameCmd;
  delete TruesOnlyCmd;
  delete SetRadialElemNbCmd;
  delete RawOutputCmd;
  delete SetTangCrystalResolCmd;
  delete SetAxialCrystalResolCmd;
  delete SetInputDataCmd;
}

void GateToSinoAccelMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if (command == SetFileNameCmd)
    { m_gateToSinoAccel->SetFileName(newValue); }
  else if ( command==TruesOnlyCmd )
    { m_gateToSinoAccel->TruesOnly(TruesOnlyCmd->GetNewBoolValue(newValue)) ; }
  else if ( command==SetRadialElemNbCmd )
    { m_gateToSinoAccel->SetRadialElemNb(SetRadialElemNbCmd->GetNewIntValue(newValue)); }
  else if ( command==RawOutputCmd )
    { m_gateToSinoAccel->RawOutputEnable(RawOutputCmd->GetNewBoolValue(newValue)); }
  else if ( command==SetTangCrystalResolCmd )
    { m_gateToSinoAccel->SetTangCrystalResolution(SetTangCrystalResolCmd->GetNewDoubleValue(newValue)); }
  else if ( command==SetAxialCrystalResolCmd )
    { m_gateToSinoAccel->SetAxialCrystalResolution(SetAxialCrystalResolCmd->GetNewDoubleValue(newValue)); }
  else if (command == SetInputDataCmd)
    { m_gateToSinoAccel->SetOutputDataName(newValue); }
  else
    { GateOutputModuleMessenger::SetNewValue(command,newValue); }
}
