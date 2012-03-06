/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateSourceGPUVoxellized.hh"
#include "GateClock.hh"
#include "Randomize.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"

#include "G4Gamma.hh"
#include "G4GenericIon.hh"
#include "G4Event.hh"
#include "G4UnitsTable.hh"

#include <vector>
#include <map>
#include "GateSourceGPUVoxellizedMessenger.hh"
#include "GateVSourceVoxelReader.hh"
#include "GateSourceVoxelTestReader.hh"
#include "GateSourceVoxelImageReader.hh"
#include "GateSourceVoxelInterfileReader.hh"

//-------------------------------------------------------------------------------------------------
GateSourceGPUVoxellized::GateSourceGPUVoxellized(G4String name)
  : GateSourceVoxellized(name)
{
  G4cout << "GateSourceGPUVoxellizedMessenger constructor" << G4endl;
  m_sourceVoxellizedMessenger = new GateSourceGPUVoxellizedMessenger(this);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
GateSourceGPUVoxellized::~GateSourceGPUVoxellized()
{
  G4cout << "GateSourceGPUVoxellizedMessenger destructor" << G4endl;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
G4double GateSourceGPUVoxellized::GetNextTime(G4double timeNow)
{
  G4cout << "GateSourceGPUVoxellizedMessenger GetNextTime" << G4endl;
  return GateSourceVoxellized::GetNextTime(timeNow);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSourceGPUVoxellized::Dump(G4int level) 
{
  G4cout << "GateSourceGPUVoxellizedMessenger Dump" << G4endl;
  return GateSourceVoxellized::Dump(level);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
G4int GateSourceGPUVoxellized::GeneratePrimaries(G4Event* event) 
{
  G4cout << "GateSourceGPUVoxellizedMessenger GeneratePrimaries" << G4endl;
  return GateSourceVoxellized::GeneratePrimaries(event);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSourceGPUVoxellized::ReaderInsert(G4String readerType)
{
  G4cout << "GateSourceGPUVoxellizedMessenger ReaderInsert" << G4endl;
  return GateSourceVoxellized::ReaderInsert(readerType);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSourceGPUVoxellized::ReaderRemove()
{
  G4cout << "GateSourceGPUVoxellizedMessenger ReaderRemove" << G4endl;
  return GateSourceVoxellized::ReaderRemove();
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSourceGPUVoxellized::Update()
{
  G4cout << "GateSourceGPUVoxellizedMessenger Update" << G4endl;
  return GateSourceVoxellized::Update();
}
//-------------------------------------------------------------------------------------------------

