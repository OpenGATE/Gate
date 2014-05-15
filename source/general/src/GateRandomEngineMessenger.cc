/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateRandomEngineMessenger.hh"
#include "GateRandomEngine.hh"
#include "GateMessenger.hh"
#include "CLHEP/Random/RandomEngine.h"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithoutParameter.hh"

///////////////////
//  Constructor  //
///////////////////

//!< Constructor
GateRandomEngineMessenger::GateRandomEngineMessenger(GateRandomEngine* gateRandomEngine)
: GateMessenger("random",true), m_gateRandomEngine(gateRandomEngine)
{
  //!< Command names
  G4String  cmdEngineName = GetDirectoryName()+"setEngineName";
  G4String  cmdEngineSeed = GetDirectoryName()+"setEngineSeed";
  G4String  cmdEngineVerbose = GetDirectoryName()+"verbose";
  G4String  cmdEngineShowStatus = GetDirectoryName()+"showStatus";
  G4String  cmdEngineFromFile = GetDirectoryName()+"resetEngineFrom"; //TC
  //!< Set the G4UI commands
  GetEngineNameCmd = new G4UIcmdWithAString(cmdEngineName,this);
  GetEngineSeedCmd = new G4UIcmdWithAString(cmdEngineSeed,this);
  GetEngineVerboseCmd = new G4UIcmdWithAnInteger(cmdEngineVerbose,this);
  ShowEngineStatus = new G4UIcmdWithoutParameter(cmdEngineShowStatus,this);
  GetEngineFromFileCmd = new G4UIcmdWithAString(cmdEngineFromFile,this); //TC
  //!< Set the guidance for those G4UI commands
  GetEngineNameCmd->SetGuidance("Set the type of the random engine");
  G4String seedGuidance = "Set the seed of the random engine:\n   - default (set the seed to the default CLHEP internal value, always the same)\n   - auto (the seed is automatically and randomly generated using the CPU time and the process ID of the Gate instance)\n   - aValue (the seed is manually set by the users, just give a long unsigned int included in [0,900000000])";
  GetEngineSeedCmd->SetGuidance(seedGuidance);
  GetEngineVerboseCmd->SetGuidance("Set the verbosity of the random engine, from 0 to 2:\n   - 0 is quiet\n   - 1 is printing one time at the beggining of the acquisition\n   - 2 is printing at each beginning of run");
  GetEngineFromFileCmd->SetGuidance("Set the seed from a file. Specify the entire path of the file"); //TC
  ShowEngineStatus->SetGuidance("Dump random engine status");
}

//////////////////
//  Destructor  //
//////////////////

//!< Destructor
GateRandomEngineMessenger::~GateRandomEngineMessenger()
{
  delete GetEngineNameCmd;
  delete GetEngineSeedCmd;
  delete GetEngineVerboseCmd;
  delete GetEngineFromFileCmd; //TC
  delete ShowEngineStatus;
}

///////////////////
//  SetNewValue  //
///////////////////

//!< void SetNewValue
void GateRandomEngineMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if( command == GetEngineNameCmd )
    { m_gateRandomEngine->SetRandomEngine(newValue); }
  else if( command == GetEngineSeedCmd )
    { m_gateRandomEngine->SetEngineSeed(newValue); }
  else if( command == GetEngineVerboseCmd )
    { m_gateRandomEngine->SetVerbosity(GetEngineVerboseCmd->GetNewIntValue(newValue)); }
  else if(command == GetEngineFromFileCmd) //TC
    { m_gateRandomEngine->resetEngineFrom(newValue); } //TC
  else if(command == ShowEngineStatus)
    { m_gateRandomEngine->ShowStatus(); }
}
