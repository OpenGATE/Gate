/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#ifndef GATEVSOURCEVOXELREADER_H
#define GATEVSOURCEVOXELREADER_H 1

#include <vector>
#include <map>
#include "globals.hh"
#include "G4ThreeVector.hh"

class GateVSource;
class GateVSourceVoxelTranslator;

//-------------------------------------------------------------------------------------------------
class GateVSourceVoxelReader
{
public:

  GateVSourceVoxelReader(GateVSource* source);

  virtual ~GateVSourceVoxelReader();

  virtual void ReadFile(G4String fileName) = 0;

  /** It is used internally by PrepareNextEvent
   * to decide which source has to be used for the current event.
   */
  virtual G4int GetNextSource();

  GateVSource* GetSource() { return m_source; };

  G4String GetName()      { return m_name; };

  virtual G4double GetTotalActivity() { return m_activityTotal; }

  virtual G4double GetTempTotalActivity() { return m_tactivityTotal; }

  virtual void SetTempTotalActivity(G4double value) { m_tactivityTotal = value; }

  virtual void SetVerboseLevel(G4int value) { nVerboseLevel = value; }

  virtual void AddVoxel(G4int ix, G4int iy, G4int iz, G4double activity);

  void SetTimeActivTables( G4String );

  void SetTimeSampling ( G4double );

  G4double GetTimeSampling ();

  virtual void ReadRTFile(G4String header_fileName, G4String fileName) = 0;

  void UpdateActivities();

  virtual void Initialize();

  void UpdateActivities(G4String,G4String);


  G4ThreeVector ComputeSourcePositionFromIsoCenter(G4ThreeVector p);

  virtual void          SetVoxelSize(G4ThreeVector size) { m_voxelSize = size; };
  virtual G4ThreeVector GetVoxelSize()                   { return m_voxelSize; };
  virtual void 			SetArraySize(G4ThreeVector arraySize) { m_voxelNx = arraySize[0]; m_voxelNy = arraySize[1]; m_voxelNz = arraySize[2]; m_sourceVoxelActivities.resize(m_voxelNx*m_voxelNy*m_voxelNz); }

  virtual void          SetPosition(G4ThreeVector pos) { m_position = pos; };
  virtual G4ThreeVector GetPosition()                  { return m_position; };

  void InsertTranslator(G4String translatorType);
  void RemoveTranslator();

  virtual void Dump(G4int level);
  void ExportSourceActivityImage(G4String activityImageFileName);

  typedef std::vector<G4double> GateSourceActivityMap;

  GateSourceActivityMap GetSourceActivityMap() { return m_sourceVoxelActivities; }

protected:
  G4int nVerboseLevel;
  G4String                       m_name;
  G4String                       m_fileName;
  GateVSource*                   m_source;
  typedef std::map<G4double,G4int >  GateSourceIntegratedActivityMap;
  GateSourceActivityMap           m_sourceVoxelActivities;
  GateSourceIntegratedActivityMap m_sourceVoxelIntegratedActivities;
  void PrepareIntegratedActivityMap();
  G4ThreeVector                  m_voxelSize;
  G4int							 m_voxelNx;
  G4int							 m_voxelNy;
  G4int							 m_voxelNz;
  G4ThreeVector                  m_position;
  G4double                       m_activityTotal;
  G4double                       m_tactivityTotal;
  G4String                       m_type;
  GateVSourceVoxelTranslator*    m_voxelTranslator;
  G4ThreeVector                  m_image_origin;
  G4double m_TS; // time sampling for time dependent activities
  G4int cK;
  G4bool IsFirstTime;
  std::map< std::pair<G4double,G4double> , std::vector<std::pair<G4double,G4double> >  > m_TimeActivTables; // for time activity curves
public:
  inline G4int RealArrayIndex(G4int ix, G4int iy, G4int iz) const
  {
	  return ix + iy*m_voxelNx + iz*m_voxelNx*m_voxelNy;
  }
  inline G4ThreeVector GetVoxelIndices(G4int arrayIndex) {
	  return G4ThreeVector( arrayIndex % m_voxelNx, (arrayIndex / m_voxelNx) % m_voxelNy, arrayIndex / (m_voxelNx*m_voxelNy));
  }

};
//-------------------------------------------------------------------------------------------------

#endif
