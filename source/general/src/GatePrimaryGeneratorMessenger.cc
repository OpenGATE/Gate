/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GatePrimaryGeneratorMessenger.hh"

#include "GateClock.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "GatePrimaryGeneratorAction.hh"

//-------------------------------------------------------------------------------------------------
GatePrimaryGeneratorMessenger::GatePrimaryGeneratorMessenger(GatePrimaryGeneratorAction* primGen)
{ 
  m_primaryGenerator = primGen;

  GateGeneratorDir = new G4UIdirectory("/gate/generator/");
  GateGeneratorDir->SetGuidance("GATE event generator control.");

  VerboseCmd = new G4UIcmdWithAnInteger("/gate/generator/verbose",this);
  VerboseCmd->SetGuidance("Set GATE event action verbose level");
  VerboseCmd->SetParameterName("verbose",false);
  VerboseCmd->SetRange("verbose>=0");

  GPSCmd = new G4UIcmdWithoutParameter("/gate/EnableGeneralParticleSource", this);
  GPSCmd->SetGuidance("Allow GATE to use /gps commands (but no other GATE sources)");
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
GatePrimaryGeneratorMessenger::~GatePrimaryGeneratorMessenger()
{
  delete VerboseCmd;
  delete GateGeneratorDir;
  delete GPSCmd;
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
void GatePrimaryGeneratorMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if (command == VerboseCmd) m_primaryGenerator->SetVerboseLevel(VerboseCmd->GetNewIntValue(newValue));
  if (command == GPSCmd) m_primaryGenerator->EnableGPS(true);
}
//-------------------------------------------------------------------------------------------------

