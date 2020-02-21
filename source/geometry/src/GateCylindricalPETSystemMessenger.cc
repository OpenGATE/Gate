/*----------------------
  OpenGATE Collaboration

  Daniel Strul <daniel.strul@iphe.unil.ch>

  Copyright (C) 2002,2003 UNIL/IPHE, CH-1015 Lausanne

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateCylindricalPETSystemMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

#include "GateCylindricalPETSystem.hh"

// Constructor
// The flags are passed to the base-class GateNamedObjectMessenger
GateCylindricalPETSystemMessenger::GateCylindricalPETSystemMessenger(GateCylindricalPETSystem* itsCylindricalPETSystem,
                                                                     const G4String& itsDirectoryName)
  : GateClockDependentMessenger(itsCylindricalPETSystem,itsDirectoryName)
{
  SetDirectoryGuidance(G4String("Controls the system '") + itsCylindricalPETSystem->GetObjectName() + "'" );
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName()+"addAnewRsector";
  addNewRsectorcmd = new G4UIcmdWithAString(cmdName,this);
}


GateCylindricalPETSystemMessenger::~GateCylindricalPETSystemMessenger()
{
  delete addNewRsectorcmd ;
}


// UI command interpreter method
void GateCylindricalPETSystemMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if ( command == addNewRsectorcmd ) GetCylindricalPETSystem()->AddNewRSECTOR( newValue );
  GateNamedObjectMessenger::SetNewValue(command,newValue);
}
