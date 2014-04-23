/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVSourceVoxelReader_h
#define GateVSourceVoxelReader_h 1

#include "globals.hh"
#include "G4ThreeVector.hh"
#include <vector>
#include <map>

class GateVSource;
class GateVSourceVoxelTranslator;

class GateVSourceVoxelReader
{
public:

  GateVSourceVoxelReader(GateVSource* source);
  virtual ~GateVSourceVoxelReader();

  virtual void ReadFile(G4String fileName) = 0;

  /** It is used internally by PrepareNextEvent
   * to decide which source has to be used for the current event.
   */
  virtual std::vector<G4int> GetNextSource();

  GateVSource* GetSource() { return m_source; };

  G4String GetName()      { return m_name; };

  virtual G4double GetTotalActivity() { return m_activityTotal; };

  virtual G4double GetTempTotalActivity() { return m_tactivityTotal; };  // added by I. Martinez-Rovira (immamartinez@gmail.com)
    
  virtual void SetTempTotalActivity(G4double value) { m_tactivityTotal = value; };  // added by I. Martinez-Rovira (immamartinez@gmail.com)

  virtual void SetVerboseLevel(G4int value) { nVerboseLevel = value; };

  virtual void AddVoxel(G4int ix, G4int iy, G4int iz, G4double activity);
  
  /* PY Descourt 08/09/2009*/
  virtual void AddVoxel_FAST(G4int, G4int, G4int, G4double);
  void SetTimeActivTables( G4String );
  void SetTimeSampling ( G4double );
  G4double GetTimeSampling ();
  virtual void ReadRTFile(G4String header_fileName, G4String fileName) = 0;

  void UpdateActivities();
  
  virtual void Initialize();
  
  void UpdateActivities(G4String,G4String);
  /* PY Descourt 08/09/2009*/
  
  
  virtual void          SetVoxelSize(G4ThreeVector size) { m_voxelSize = size; };
  virtual G4ThreeVector GetVoxelSize()                   { return m_voxelSize; };

  virtual void          SetPosition(G4ThreeVector pos) { m_position = pos; };
  virtual G4ThreeVector GetPosition()                  { return m_position; };

  void InsertTranslator(G4String translatorType);
  void RemoveTranslator();

  virtual void Dump(G4int level);

  typedef std::map<std::vector<G4int>,G4double>   GateSourceActivityMap;
  GateSourceActivityMap GetSourceActivityMap() { return m_sourceVoxelActivities; }
protected:
  G4int nVerboseLevel;

  G4String                       m_name;
  G4String                       m_fileName;

  GateVSource*                    m_source;

  std::vector<G4int>                  m_firstSource;
  
  typedef std::map<G4double,std::vector<G4int> >  GateSourceIntegratedActivityMap;
  GateSourceActivityMap           m_sourceVoxelActivities;
  GateSourceIntegratedActivityMap m_sourceVoxelIntegratedActivities;

  void PrepareIntegratedActivityMap();

  G4ThreeVector                  m_voxelSize;
  G4ThreeVector                  m_position;

  G4double                       m_activityMax;
  G4double                       m_activityTotal;
  G4double                       m_tactivityTotal;  // added by I. Martinez-Rovira (immamartinez@gmail.com)

  G4String                       m_type;

  GateVSourceVoxelTranslator*    m_voxelTranslator;
  
  /* PY Descourt 08/09/2009 */

  G4double m_TS; // time sampling for time dependent activities
  G4int cK;
  G4bool IsFirstTime;
  std::map< std::pair<G4double,G4double> , std::vector<std::pair<G4double,G4double> >  > m_TimeActivTables; // for time activity curves  
  /* PY Descourt 08/09/2008 */
};

#endif


