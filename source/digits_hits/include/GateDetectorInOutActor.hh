/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateDetectorInOutActor
*/

#include "GateConfiguration.h"

#ifndef GATEDETECTORINOUT_HH
#define GATEDETECTORINOUT_HH

#include "GateActorManager.hh"
#include "GateMiscFunctions.hh"
#include "GateVActor.hh"
#include "GateDetectorInOutActorMessenger.hh"


//-----------------------------------------------------------------------------
struct DetectorInData {
  // Input parameters
  double x;     // in mm
  double y;     // in mm
  double theta; // radian along X
  double phi;   // radian along Y
  double E;     // in MeV
  // Helper
  void Print(std::ostream & os);
};
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
struct DetectorOutData {
  // Output parameters
  //double u;
  //double v;
  double w;     // windows id (0 if outside)
  // Helper
  void Print(std::ostream & os);
};
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
class GateDetectorInOutActor: public GateVActor
{
public:

  // Macro to auto declare actor
  FCT_FOR_AUTO_CREATOR_ACTOR(GateDetectorInOutActor)

  // Actor name
  virtual ~GateDetectorInOutActor();

  // Constructs the sensor
  virtual void Construct();

  // Parameters
  void SetOutputWindowNames(std::string & names);
  void SetOutputInDataOnlyFlag(bool b);
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
  GateDetectorInOutActor(G4String name, G4int depth = 0);
  GateDetectorInOutActorMessenger * pMessenger;

  bool mOutputInDataOnlyFlag;
  bool mIgnoreCurrentData;
  std::vector<DetectorOutData> mOutData;
  std::vector<DetectorInData> mInData;
  bool mEventIsAlreadyStored;
  DetectorOutData mCurrentOutData;
  DetectorInData mCurrentInData;
  std::vector<G4String> mListOfWindowNames;
  std::vector<int> mListOfWindowIds;
  int mNumberOfDetectedEvent;
  int mRRFactor;
  double mMaxAngle;
  double mThetaMax;
  double mPhiMax;
};

// Macro to auto declare actor
MAKE_AUTO_CREATOR_ACTOR(DetectorInOutActor, GateDetectorInOutActor)

#endif /* end #define GATEDETECTORINOUT_HH */
