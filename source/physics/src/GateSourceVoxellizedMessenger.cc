/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSourceVoxellizedMessenger.hh"
#include "GateSourceVoxellized.hh"

#include "GateClock.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateSourceVoxellizedMessenger::GateSourceVoxellizedMessenger(GateSourceVoxellized* source)
  : GateMessenger(G4String("source/") + source->GetName(), false),
    m_source(source)
{ 

  G4String cmdName;

  cmdName = GetDirectoryName()+"reader/insert";
  ReaderInsertCmd = new GateUIcmdWithAVector<G4String>(cmdName,this);
  ReaderInsertCmd->SetGuidance("Insert a reader");
  ReaderInsertCmd->SetGuidance("1. Reader type");

  cmdName = GetDirectoryName()+"reader/remove";
  ReaderRemoveCmd = new G4UIcmdWithoutParameter(cmdName,this);
  ReaderRemoveCmd->SetGuidance("Remove the reader");

  cmdName = GetDirectoryName()+"setPosition";
  PositionCmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
  PositionCmd->SetGuidance("Set the source global position");
  PositionCmd->SetGuidance("1. Position xyz and unit");

}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateSourceVoxellizedMessenger::~GateSourceVoxellizedMessenger()
{
   delete PositionCmd;
   delete ReaderInsertCmd;
   delete ReaderRemoveCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateSourceVoxellizedMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command == ReaderInsertCmd ) {
    m_source->ReaderInsert(ReaderInsertCmd->GetNewVectorValue(newValue)[0]);
  } else if( command == ReaderRemoveCmd ) {
    m_source->ReaderRemove();
  } else if( command == PositionCmd ) {
    m_source->SetPosition(PositionCmd->GetNew3VectorValue(newValue));
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....



