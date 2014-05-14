/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateSourceVoxellized.hh"
#include "GateClock.hh"
#include "Randomize.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"

#include "G4Gamma.hh"
// vesna
#include "G4OpticalPhoton.hh"
// vesna

#include "G4GenericIon.hh"
#include "G4Event.hh"
#include "G4UnitsTable.hh"

#include <vector>
#include <map>
#include "GateSourceVoxellizedMessenger.hh"
#include "GateVSourceVoxelReader.hh"
#include "GateSourceVoxelTestReader.hh"
#include "GateSourceVoxelImageReader.hh"
#include "GateSourceVoxelInterfileReader.hh"

//-------------------------------------------------------------------------------------------------
GateSourceVoxellized::GateSourceVoxellized(G4String name)
  : GateVSource(name) 
  , m_sourcePosition(G4ThreeVector())
  , m_sourceRotation(G4RotationMatrix())
  , m_voxelReader(0)
{
  m_sourceVoxellizedMessenger = new GateSourceVoxellizedMessenger(this);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
GateSourceVoxellized::~GateSourceVoxellized()
{
  if (nVerboseLevel > 0) 
    G4cout << "GateSourceVoxellized::~GateSourceVoxellized " << G4endl;
  delete m_sourceVoxellizedMessenger;
  if (m_voxelReader) delete m_voxelReader;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
G4double GateSourceVoxellized::GetNextTime(G4double timeNow)
{
  if (!m_voxelReader) {
    G4cout << "GateSourceVoxellized::GetNextTime: insert a voxel reader first" << G4endl;
    return 0.;
  }
  // compute random time for this source as if it was one source with the total activity
  m_activity = m_voxelReader->GetTempTotalActivity();  // modified by I. Martinez-Rovira (immamartinez@gmail.com)
  G4double firstTime = GateVSource::GetNextTime(timeNow); 

  if (nVerboseLevel>1) 
    G4cout << "GateSourceVoxellized::GetNextSource : firstTime (s) " << firstTime/s << G4endl;

  return firstTime;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSourceVoxellized::Dump(G4int level) 
{
  G4cout << "Source ---------------> " << m_name << G4endl
	 << "  ID                  : " << m_sourceID << G4endl
	 << "  type                : " << m_type << G4endl
	 << "  startTime (s)       : " << m_startTime/s << G4endl
	 << "  time (s)            : " << m_time/s << G4endl
	 << "  forcedUnstable      : " << m_forcedUnstableFlag << G4endl;
  if ( m_forcedUnstableFlag ) 
    G4cout << "  forcedLifetime (s)  : " << m_forcedLifeTime/s << G4endl;
  G4cout << "  verboseLevel        : " << nVerboseLevel << G4endl
 	 << "----------------------- " << G4endl;
  
  if (!m_voxelReader) {
    G4cout << "GateSourceVoxellized::Dump: voxel reader not defined" << G4endl;
  } else {
    if (level > 0)
      m_voxelReader->Dump(1);
  }

  if (level > 0)
    GateVSource::Dump(1);

}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
G4int GateSourceVoxellized::GeneratePrimaries(G4Event* event) 
{
  if (!m_voxelReader) {
    G4cout << "GateSourceVoxellized::GeneratePrimaries: insert a voxel reader first" << G4endl;
    return 0;
  }
  // ask to the voxel reader to provide the active voxel for this event
  std::vector<G4int> firstSource = m_voxelReader->GetNextSource();

  // move the centre to the chosen voxel:
  // to the relative position and then to the global absolute position taking into account 
  // the rotation of the voxel matrix both for the position and for the orientation of the active voxel

  G4ThreeVector voxelSize = m_voxelReader->GetVoxelSize();
  // offset of the centre of the selected voxel wrt the matrix corner (as the (0,0,0) voxel is in the corner)
  G4ThreeVector relativeVoxelOffset = G4ThreeVector( voxelSize.x()/2. + voxelSize.x() * firstSource[0], 
						     voxelSize.y()/2. + voxelSize.y() * firstSource[1], 
						     voxelSize.z()/2. + voxelSize.z() * firstSource[2]);

  // m_sourcePosition and m_sourceRotation are NOT the ones in GPS, on the contrary they are used to set the 
  // GPS position and "position rotation" (for the moment not the "direction rotation")
  G4ThreeVector centre = m_sourcePosition + m_sourceRotation(relativeVoxelOffset);

  // rotation of the Para shape according to the rotation of the voxel matrix
  GetPosDist()->SetPosRot1(m_sourceRotation(G4ThreeVector(1.,0.,0.))); // x'
  GetPosDist()->SetPosRot2(m_sourceRotation(G4ThreeVector(0.,1.,0.))); // y'

  if (nVerboseLevel > 1) 
    G4cout << "[GateSourceVoxellized::GeneratePrimaries] Centre: " << G4BestUnit(centre,"Length") << G4endl;


  GetPosDist()->SetCentreCoords(centre);
  GetPosDist()->SetPosDisType("Volume");
  GetPosDist()->SetPosDisShape("Para");
  GetPosDist()->SetHalfX(voxelSize.x()/2.);
  GetPosDist()->SetHalfY(voxelSize.y()/2.);
  GetPosDist()->SetHalfZ(voxelSize.z()/2.);

  // shoot the primary
  G4int numVertices = GateVSource::GeneratePrimaries(event);

  return numVertices;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSourceVoxellized::ReaderInsert(G4String readerType)
{
  if (m_voxelReader) {
    G4cout << "GateSourceVoxellized::ReaderInsert: voxel reader already defined" << G4endl;
  } else {
    if (readerType == G4String("test")) {
      m_voxelReader = new GateSourceVoxelTestReader(this);
    } else if (readerType == G4String("image")) {
      m_voxelReader = new GateSourceVoxelImageReader(this);
    } else if (readerType == G4String("interfile")) {
      m_voxelReader = new GateSourceVoxelInterfileReader(this);
    }
    else {
      G4cout << "GateSourceVoxellized::ReaderInsert: unknown reader type" << G4endl;
    }
  }

}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSourceVoxellized::ReaderRemove()
{

  if (m_voxelReader) {
    delete m_voxelReader;
    m_voxelReader = 0;
  } else {
    G4cout << "GateSourceVoxellized::ReaderRemove: voxel reader not defined" << G4endl;
  }

}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSourceVoxellized::Update(G4double time)
{
  // insert here any specific update needed for the voxel matrix source, otherwise for the standard updates
  // the GateVSource::Update() is called
  if (false) {

    // this will be applied only in the case the source is attached to a volume, thus it has to be aligned to
    // the actual volume position

    m_sourceRotation = G4RotationMatrix(); // for the moment set to unity

    // position of the volume center. This is obtained from the geometry.
    G4ThreeVector volumePosition       = G4ThreeVector(); // for the moment set to zero

    // offset of the matrix corner wrt the matrix centre. This is obtained from the geometry, it's given by  
    // the half dimensions of the geometry voxel matrix. It is not a source property, as in the source there 
    // is no concept of overall dimensions, like nx ny nz, just a set of dispersed voxels)
    // WRT the value given by the box dimensions, the offset has to be rotated according to the box orientation.
    G4ThreeVector relativeCornerOffset = m_sourceRotation(G4ThreeVector()); // for the moment set to zero

    // the source position for the case of the voxels is always the "corner" of the box 
    m_sourcePosition = volumePosition - relativeCornerOffset; // attention to the "-" sign...

  } else {
    // the position is set through the source command
  }

  GateVSource::Update(time);
}
//-------------------------------------------------------------------------------------------------
