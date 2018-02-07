/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateSourceVoxellizedMessenger.hh"
#include "GateSourceVoxellized.hh"

#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//-----------------------------------------------------------------------------
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

  cmdName = GetDirectoryName()+"TranslateTheSourceAtThisIsoCenter";
  translateIsoCenterCmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
  translateIsoCenterCmd->SetGuidance("Set the source position so that the given position is at world 0,0,0");
  translateIsoCenterCmd->SetUnitCategory("Length");

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateSourceVoxellizedMessenger::~GateSourceVoxellizedMessenger()
{
   delete PositionCmd;
   delete ReaderInsertCmd;
   delete ReaderRemoveCmd;
   delete translateIsoCenterCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSourceVoxellizedMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if (command == ReaderInsertCmd)  m_source->ReaderInsert(ReaderInsertCmd->GetNewVectorValue(newValue)[0]);
  if (command == ReaderRemoveCmd)  m_source->ReaderRemove();
  if (command == PositionCmd) m_source->SetPosition(PositionCmd->GetNew3VectorValue(newValue));
  if (command == translateIsoCenterCmd) m_source->SetIsoCenterPosition(translateIsoCenterCmd->GetNew3VectorValue(newValue));
  GateMessenger::SetNewValue(command, newValue);
}
//-----------------------------------------------------------------------------
