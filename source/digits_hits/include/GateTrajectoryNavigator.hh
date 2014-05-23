/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateTrajectoryNavigator_H
#define GateTrajectoryNavigator_H

#include "globals.hh"
#include "G4ThreeVector.hh"
#include <vector>

class G4TrajectoryContainer;
class G4Trajectory;


class GateTrajectoryNavigator
{
public:

  GateTrajectoryNavigator();

  virtual ~GateTrajectoryNavigator();


  G4int         FindSourceIndex();

  G4ThreeVector FindSourcePosition();

  G4int         FindPositronTrackID();

  std::vector<G4int> FindAnnihilationGammasTrackID();

  G4int         FindPhotonID(G4int trackID);

  G4int         FindPrimaryID(G4int trackID);

  void          Initialize();

  void                          SetTrajectoryContainer(G4TrajectoryContainer* trajectoryContainer);

  inline G4TrajectoryContainer* GetTrajectoryContainer()   { return m_trajectoryContainer; };

  inline std::vector<G4int>          GetPhotonIDVec()           { return m_photonIDVec; };

  inline G4int                  GetPositronTrackID()       { return m_positronTrackID; };

  void                          SetIonID();

  void                          SetVerboseLevel(G4int val) { nVerboseLevel = val; };

protected:

private:
  G4TrajectoryContainer* m_trajectoryContainer;

  std::vector<G4int>          m_photonIDVec;
  G4int                  m_positronTrackID;
  G4Trajectory*          m_positronTrj;
  G4int                  m_ionID;

  G4int                  nVerboseLevel;
};


#endif
