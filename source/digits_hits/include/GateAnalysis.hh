/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/



//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#ifndef GateAnalysis_H
#define GateAnalysis_H

#include "GateVOutputModule.hh"


class GateTrajectoryNavigator;
class GateAnalysisMessenger;
class GateVVolume;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateAnalysis :  public GateVOutputModule
{
public:

  GateAnalysis(const G4String& name, GateOutputMgr* outputMgr,DigiMode digiMode);
  virtual ~GateAnalysis();
  const G4String& GiveNameOfFile();

  void RecordBeginOfAcquisition();
  void RecordEndOfAcquisition();
  void RecordBeginOfRun(const G4Run * );
  void RecordEndOfRun(const G4Run * );
  void RecordBeginOfEvent(const G4Event * );

  //! Here we do the first analysis of the events
  /*! It reconstruct the tree of the trajectories with the GateTrajectoryNavigator
    and completes the hits information with global event info like the ID of the photon
    that generated the hit, or the number of compton diffusions of this photon.
    The source ID and source position are also retrieved and stored in the hits
    \sa GateTrajectoryNavigator
  */
  void RecordEndOfEvent(const G4Event * );

  void RecordStepWithVolume(const GateVVolume * v, const G4Step * );

  //! saves the geometry voxel information
  void RecordVoxels(GateVGeometryVoxelStore *) {};

  virtual void SetVerboseLevel(G4int val);

  // HDS : septal penetration
  inline const G4String& GetSeptalPhysVolumeName() const { return m_septalPhysVolumeName; }
  inline G4bool GetRecordSeptalFlag() const { return m_recordSeptalFlag; }
  inline void SetSeptalPhysVolumeName(const G4String& name) { m_septalPhysVolumeName = name; }
  inline void SetRecordSeptalFlag(G4bool flag) { m_recordSeptalFlag = flag; }

private:


  GateTrajectoryNavigator* m_trajectoryNavigator;

  GateAnalysisMessenger* m_analysisMessenger;
  G4String m_noFileName;
  // HDS : septal penetration
  G4String m_septalPhysVolumeName;
  G4bool m_recordSeptalFlag;

};

#endif
