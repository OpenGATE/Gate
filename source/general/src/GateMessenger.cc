/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

#include "GateDetectorConstruction.hh"


//-------------------------------------------------------------------------------------
// Constructor
// 'itsName' is the base-name of the directory
// The flag 'flagCreateDirectory' tells whether it should create a new UI directory
// (set this flag to false if the directory is already created by another messenger)
GateMessenger::GateMessenger(const G4String& itsName,G4bool createDirectory)
: mName(itsName), 
  mDirName(ComputeDirectoryName(itsName)),
  pDir(0)
{ 
  if (createDirectory){
    pDir = new G4UIdirectory(mDirName);
    G4String name;
    name = "GATE " + mName + " control.";
    pDir->SetGuidance(name);
    }
}
//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
GateMessenger::~GateMessenger()
{
  if (pDir)
    delete pDir;
}
//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
// UI command interpreter method
void GateMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
    G4UImessenger::SetNewValue(command,newValue);
}
//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
// Tells GATE that the geometrical parameters have undergone a minor modification
// so that the geometry should be updated 
/*
void GateMessenger::TellGeometryToUpdate()
{

  GateDetectorConstruction::GetGateDetectorConstruction()
    ->GeometryHasChanged(GateDetectorConstruction::geometry_needs_update);    
} 
*/ 
//-------------------------------------------------------------------------------------    


//-------------------------------------------------------------------------------------
// Tells GATE that the geometrical parameters have undergone a major modification
// so that the geometry should be rebuilt 
/*
void GateMessenger::TellGeometryToRebuild()
{

  GateDetectorConstruction::GetGateDetectorConstruction()
    ->GeometryHasChanged(GateDetectorConstruction::geometry_needs_rebuild);

} 
*/ 
//-------------------------------------------------------------------------------------

    
//-------------------------------------------------------------------------------------
// Adds a new line to the directory guidance
void GateMessenger::SetDirectoryGuidance(const G4String& guidance)
{
  GetDirectory()->SetGuidance(guidance);
}
//-------------------------------------------------------------------------------------

