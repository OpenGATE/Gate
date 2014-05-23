/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATETIMEACTOR_HH
#define GATETIMEACTOR_HH

#include "GateVActor.hh"
#include "GateActorManager.hh"
#include "GateActorMessenger.hh"
#include "GateTimeActorMessenger.hh"

#include "G4Timer.hh"

//-----------------------------------------------------------------------------
/// \brief Actor for timing steps

class GateTimeActorMessenger;
class GateTimeActor : public GateVActor
{
public:

  virtual ~GateTimeActor();

  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateTimeActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  //-----------------------------------------------------------------------------
  // Callbacks
  virtual void BeginOfRunAction(const G4Run*);
  virtual void EndOfRunAction(const G4Run*);
  virtual void BeginOfEventAction(const G4Event*);
  virtual void EndOfEventAction(const G4Event*);
  virtual void PreUserTrackingAction(const GateVVolume *, const G4Track*);
  virtual void PostUserTrackingAction(const GateVVolume *, const G4Track*);
  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);

  //-----------------------------------------------------------------------------
  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  void EnableDetailedStats(bool b);

protected:
  GateTimeActor(G4String name, G4int depth=0);

  G4Timer mCurrentRunTimer;
  G4Timer mCurrentEventTimer;
  G4Timer mCurrentTrackTimer;
  G4Timer mCurrentStepTimer;
  double mTotalEventUserTime;
  double mTotalTrackUserTime;
  double mTotalStepUserTime;
  long mNumberOfEvents;
  long mNumberOfTracks;
  long mNumberOfSteps;
  bool mDetailedStatFlag;

  typedef std::map<G4String, double> MapType;
  MapType mTimePerParticle;
  MapType mTrackPerParticle;

  G4String mCurrentParticleName;
  MapType mNumberOfLimitingProcess;
  MapType mNumberOfAlongByProcess;
  //MapType mNumberOfPostByProcess;

  void UpdateCurrentTextOutput();
  std::string mCurrentTextOutput;

  GateTimeActorMessenger * pMessenger;
};

MAKE_AUTO_CREATOR_ACTOR(TimeActor,GateTimeActor)


#endif /* end #define GATESIMULATIONSTATISTICACTOR_HH */
