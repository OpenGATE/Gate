/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
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

  cmdName = GetDirectoryName()+"setRmax";
  setRmaxCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  setRmaxCmd->SetGuidance("set the value of R");
  setRmaxCmd->SetParameterName("R value",false);
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
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateSourcePhaseSpaceMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{ 
  GateVSourceMessenger::SetNewValue(command,newValue);
  if (command == AddFileCmd ) pSource->AddFile(newValue);// {dynamic_cast<GateSourcePhaseSpace*>(m_source)->AddFile(newValue);}
  if (command == RelativeVolumeCmd) pSource->SetPositionInWorldFrame(true);// {dynamic_cast<GateSourcePhaseSpace*>(m_source)->SetPositionInWorldFrame(true);}
  if (command == RegularSymmetryCmd) pSource->SetUseRegularSymmetry();
  if (command == RandomSymmetryCmd) pSource->SetUseRandomSymmetry();
  if (command == setParticleTypeCmd) pSource->SetParticleType(newValue);
  if (command == setUseNbParticleAsIntensityCmd) 
    pSource->SetUseNbOfParticleAsIntensity(setUseNbParticleAsIntensityCmd->GetNewBoolValue(newValue));
  if(command == setRmaxCmd) pSource->SetRmax(setRmaxCmd->GetNewDoubleValue(newValue));
}
//----------------------------------------------------------------------------------------


#endif
