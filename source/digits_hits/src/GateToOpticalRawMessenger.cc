/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*! \file GateToOpticalRawMessenger.cc
   Created on   2012/07/09  by vesna.cuplov@gmail.com
   Implemented new class GateToOpticalRaw for Optical photons: write result of the projection.
*/


#include "GateToOpticalRawMessenger.hh"
#include "GateToOpticalRaw.hh"
#include "GateOutputMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

GateToOpticalRawMessenger::GateToOpticalRawMessenger(GateToOpticalRaw* gateToOpticalRaw)
  : GateOutputModuleMessenger(gateToOpticalRaw)
  , m_gateToOpticalRaw(gateToOpticalRaw)
{
  G4String cmdName;

}

GateToOpticalRawMessenger::~GateToOpticalRawMessenger()
{
//  delete SetFileNameCmd;
}

void GateToOpticalRawMessenger::SetNewValue(G4UIcommand* command,G4String /*newValue*/)
{
  // All mother macro commands are overloaded to do nothing
  if( command == GetVerboseCmd() ) {
    G4cout << "GateToOpticalRaw::VerboseCmd: Do nothing" << G4endl;
  } else if( command == GetDescribeCmd() ) {
    G4cout << "GateToOpticalRaw::DescribeCmd: Do nothing" << G4endl;
  } else if ( command == GetEnableCmd() ) {
    G4cout << "GateToOpticalRaw::EnableCmd: Do nothing" << G4endl;
  } else if ( command == GetDisableCmd() ) {
    G4cout << "GateToOpticalRaw::DisableCmd: Do nothing" << G4endl;
  }
/* No else anymore
  else
    { GateOutputModuleMessenger::SetNewValue(command,newValue);  }
*/
}
