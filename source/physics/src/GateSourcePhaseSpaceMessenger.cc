/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateSourcePhaseSpaceMessenger.hh"
#include "GateSourcePhaseSpace.hh"
#include "GateClock.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//----------------------------------------------------------------------------------------
GateSourcePhaseSpaceMessenger::GateSourcePhaseSpaceMessenger(GateSourcePhaseSpace* source)
  : GateVSourceMessenger(source),pSource(source)
{
  G4String cmdName;

  cmdName = GetDirectoryName()+"addPhaseSpaceFile";
  AddFileCmd = new G4UIcmdWithAString(cmdName,this);
  AddFileCmd->SetGuidance("Add a phase space file");
  AddFileCmd->SetParameterName("File Name",false);

  cmdName = GetDirectoryName()+"setPhaseSpaceInWorldFrame";
  RelativeVolumeCmd = new G4UIcmdWithoutParameter(cmdName,this);
  RelativeVolumeCmd->SetGuidance("Use this command if particles in the phase space are defined in the world frame.");

  cmdName = GetDirectoryName()+"useRegularSymmetry";
  RegularSymmetryCmd = new G4UIcmdWithoutParameter(cmdName,this);
  RegularSymmetryCmd->SetGuidance("Use the rotational symmetry axis of the source with a regular step.");

  cmdName = GetDirectoryName()+"useRandomSymmetry";
  RandomSymmetryCmd = new G4UIcmdWithoutParameter(cmdName,this);
  RandomSymmetryCmd->SetGuidance("Use the rotational symmetry axis of the source with a random step.");

  cmdName = GetDirectoryName()+"setParticleType";
  setParticleTypeCmd = new G4UIcmdWithAString(cmdName,this);
  setParticleTypeCmd->SetGuidance("set the particle type (if not given in the PhS)");
  setParticleTypeCmd->SetParameterName("Particle Type",false);

  cmdName = GetDirectoryName()+"useNbOfParticleAsIntensity";
  setUseNbParticleAsIntensityCmd = new G4UIcmdWithABool(cmdName,this);
  setUseNbParticleAsIntensityCmd->SetGuidance("use the nb of particle in the PhS as source intensity");

  cmdName = GetDirectoryName()+"ignoreWeight";
  ignoreWeightCmd = new G4UIcmdWithABool(cmdName,this);
  ignoreWeightCmd->SetGuidance("Force weight to 1.0 even if weight exist in the phase space");

  cmdName = GetDirectoryName()+"setRmax";
  setRmaxCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  setRmaxCmd->SetGuidance("set the value of R");
  setRmaxCmd->SetParameterName("R value",false);

  cmdName = GetDirectoryName()+"setStartingParticleId";
  setStartIdCmd = new G4UIcmdWithADouble(cmdName,this);
  setStartIdCmd->SetGuidance("set the id of the particle to start with");

  cmdName = GetDirectoryName()+"setPytorchBatchSize";
  setPytorchBatchSizeCmd = new G4UIcmdWithAnInteger(cmdName,this);
  setPytorchBatchSizeCmd->SetGuidance("set the batch size for pytorch PHSP");

  cmdName = GetDirectoryName()+"setPytorchParams";
  setPytorchParamsCmd = new G4UIcmdWithAString(cmdName,this);
  setPytorchParamsCmd->SetGuidance("set the json file associated with the .pt PHSP");
  setPytorchParamsCmd->SetParameterName("Filename",false);

  cmdName = GetDirectoryName()+"setSphereRadius";
  setSphereRadiusCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  setSphereRadiusCmd->SetGuidance("set the radius in mm of the sphere to project the particles (EXPERIMENTAL)");
  setSphereRadiusCmd->SetParameterName("Radius value",false);

}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
GateSourcePhaseSpaceMessenger::~GateSourcePhaseSpaceMessenger()
{
  delete AddFileCmd;
  delete RelativeVolumeCmd;
  delete RegularSymmetryCmd;
  delete RandomSymmetryCmd;
  delete setParticleTypeCmd;
  delete setUseNbParticleAsIntensityCmd;
  delete setRmaxCmd;
  delete setStartIdCmd;
  delete setPytorchBatchSizeCmd;
  delete setPytorchParamsCmd;
  delete setSphereRadiusCmd;
  delete ignoreWeightCmd;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateSourcePhaseSpaceMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  GateVSourceMessenger::SetNewValue(command,newValue);
  if (command == AddFileCmd ) pSource->AddFile(newValue);
  if (command == RelativeVolumeCmd) pSource->SetPositionInWorldFrame(true);
  if (command == RegularSymmetryCmd) pSource->SetUseRegularSymmetry();
  if (command == RandomSymmetryCmd) pSource->SetUseRandomSymmetry();
  if (command == setParticleTypeCmd) pSource->SetParticleType(newValue);
  if (command == setUseNbParticleAsIntensityCmd)
    pSource->SetUseNbOfParticleAsIntensity(setUseNbParticleAsIntensityCmd->GetNewBoolValue(newValue));
   if (command == ignoreWeightCmd) pSource->SetIgnoreWeight(ignoreWeightCmd->GetNewBoolValue(newValue)); 
  if (command == setRmaxCmd) pSource->SetRmax(setRmaxCmd->GetNewDoubleValue(newValue));
  if (command == setSphereRadiusCmd) pSource->SetSphereRadius(setSphereRadiusCmd->GetNewDoubleValue(newValue));
  if (command == setStartIdCmd) pSource->SetStartingParticleId(setStartIdCmd->GetNewDoubleValue(newValue));
  if (command == setPytorchBatchSizeCmd) pSource->SetPytorchBatchSize(setPytorchBatchSizeCmd->GetNewIntValue(newValue));
  if (command == setPytorchParamsCmd) pSource->SetPytorchParams(newValue);
  
}
//----------------------------------------------------------------------------------------

#endif
