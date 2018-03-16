/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  Gate_NN_ARF_Actor
*/

#include "GateConfiguration.h"

#ifndef GATE_NN_ARF_ACTOR_HH
#define GATE_NN_ARF_ACTOR_HH

#include "GateActorManager.hh"
#include "GateMiscFunctions.hh"
#include "GateVActor.hh"
#include "Gate_NN_ARF_ActorMessenger.hh"

//-----------------------------------------------------------------------------
struct Gate_NN_ARF_Train_Data {
  double theta; // in deg, angle along X
  double phi;   // in deg, angle along Y
  double E;     // in MeV
  double w;     // windows id (0 if outside)
  // Helper
  void Print(std::ostream & os);
};
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
struct Gate_NN_ARF_Test_Data {
  double x;     // in mm
  double y;     // in mm
  double theta; // in deg, angle along X
  double phi;   // in deg, angle along Y
  double E;     // in MeV
  // Helper
  void Print(std::ostream & os);
};
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
class Gate_NN_ARF_Actor: public GateVActor
{
public:

  // Macro to auto declare actor
  FCT_FOR_AUTO_CREATOR_ACTOR(Gate_NN_ARF_Actor)

  // Actor name
  virtual ~Gate_NN_ARF_Actor();

  // Constructs the sensor
  virtual void Construct();

  // Parameters
  void SetEnergyWindowNames(std::string & names);
  void SetMode(std::string m);
  void SetMaxAngle(double a);
  void SetRRFactor(int f);

  // Callbacks
  virtual void BeginOfRunAction(const G4Run *);
  virtual void BeginOfEventAction(const G4Event * e);
  virtual void EndOfEventAction(const G4Event * e);
  virtual void UserSteppingAction(const GateVVolume * v, const G4Step* step);

  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

protected:
  Gate_NN_ARF_Actor(G4String name, G4int depth = 0);
  Gate_NN_ARF_ActorMessenger * pMessenger;

  bool mTrainingModeFlag;
  bool mIgnoreCurrentData;
  std::vector<Gate_NN_ARF_Test_Data>  mTestData;
  std::vector<Gate_NN_ARF_Train_Data> mTrainData;
  bool mEventIsAlreadyStored;
  Gate_NN_ARF_Test_Data mCurrentTestData;
  Gate_NN_ARF_Train_Data mCurrentTrainData;
  std::vector<G4String> mListOfWindowNames;
  std::vector<int> mListOfWindowIds;
  int mNumberOfDetectedEvent;
  int mRRFactor;
  double mMaxAngle;
  double mThetaMax;
  double mPhiMax;
};

// Macro to auto declare actor
MAKE_AUTO_CREATOR_ACTOR(NN_ARF_Actor, Gate_NN_ARF_Actor)

#endif /* end #define GATEDETECTORINOUT_HH */
