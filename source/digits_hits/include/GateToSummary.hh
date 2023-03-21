/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GATETOSUMMARY_H
#define GATETOSUMMARY_H

#include <vector>
#include <map>

#include "GateVOutputModule.hh"

#ifdef G4ANALYSIS_USE_FILE

class GateToSummaryMessenger;

//--------------------------------------------------------------------------------
class GateToSummary :  public GateVOutputModule
{
public:
  GateToSummary(const G4String& name, GateOutputMgr* outputMgr, DigiMode digiMode);

  virtual ~GateToSummary();
  virtual const G4String& GiveNameOfFile();

  virtual void RecordBeginOfAcquisition();
  virtual void RecordEndOfAcquisition();
  virtual void RecordBeginOfRun(const G4Run *) {}
  virtual void RecordEndOfRun(const G4Run *);
  virtual void RecordBeginOfEvent(const G4Event *) {}
  virtual void RecordEndOfEvent(const G4Event *);
  virtual void RecordStepWithVolume(const GateVVolume * , const G4Step *) {}
  virtual void RecordVoxels(GateVGeometryVoxelStore *) {}

  const  G4String& GetFileName() { return m_fileName; };
  void SetFileName(const G4String aName) { m_fileName = aName; };

  void addCollection(const std::string & str);

private:

  GateToSummaryMessenger* m_summaryMessenger;
  G4String m_fileName;

  std::vector<std::string>  m_hits_collection_names;
  std::map<std::string, G4int> m_hits_nb;

  G4int m_nb_of_events;
  G4int m_nb_of_runs;

  std::vector<std::string> m_collection_names;
  std::map<std::string, G4int> m_collection_nb;
};
//--------------------------------------------------------------------------------

#endif
#endif
