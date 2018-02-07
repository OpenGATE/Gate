/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateFluenceActor
  \author simon.rit@creatis.insa-lyon.fr
*/

#include "GateConfiguration.h"

#ifndef GATEFLUENCEACTOR_HH
#define GATEFLUENCEACTOR_HH

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "GateMiscFunctions.hh"
#include "GateFluenceActorMessenger.hh"
#include "GateEnergyResponseFunctor.hh"
#include "GateImageWithStatistic.hh"

class GateFluenceActor: public GateVImageActor
{
public:


  // Actor name
  virtual ~GateFluenceActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateFluenceActor)


  // Constructs the sensor
  virtual void Construct();

  void EnableSquaredImage(bool b)
  {
    mIsSquaredImageEnabled = b;
  }
  void EnableStepLengthImage(bool b)
  {
    mIsStepLengthImageEnabled = b;
  }
  void EnableUncertaintyImage(bool b)
  {
    mIsUncertaintyImageEnabled = b;
  }
  void EnableNormalisation(bool b)
  {
    mIsNormalisationEnabled = b;
    mImage.SetScaleFactor(1.0);
  }
  void EnableNumberOfHitsImage(bool b)
  {
    mIsNumberOfHitsImageEnabled = b;
  }
  void EnableScatterImage(bool b)
  {
    mIsScatterImageEnabled = b;
  }
  void SetIgnoreWeight(bool b)
  {
    mIgnoreWeight = b;
  }
  virtual void BeginOfRunAction(const G4Run *);
  virtual void BeginOfEventAction(const G4Event * e);
  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step);
  virtual void UserPreTrackActionInVoxel(const int /*index*/,
                                         const G4Track* /*t*/)
  {
  }
  virtual void UserPostTrackActionInVoxel(const int /*index*/,
                                          const G4Track* /*aTrack*/);

  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  ///Scorer related
  //virtual G4bool ProcessHits(G4Step *, G4TouchableHistory*);
  virtual void Initialize(G4HCofThisEvent*)
  {
  }
  virtual void EndOfEvent(G4HCofThisEvent*)
  {
  }

  void SetResponseDetectorFile(G4String name)
  {
    mResponseFileName = name;
  }
  void SetScatterOrderFilename(G4String name)
  {
    mScatterOrderFilename = name;
  }
  void SetSeparateProcessFilename(G4String name)
  {
    mSeparateProcessFilename = name;
  }

protected:

  GateImageWithStatistic mImage;
  GateImageWithStatistic mImageProcess;
  GateImage mLastHitEventImage;
  GateImage mNumberOfHitsImage;
  GateImageDouble mStepLengthImage;
  GateImageDouble mNumberOfHitsStepLengthImage;

  //GateImage mImageScatter;
  GateFluenceActor(G4String name, G4int depth = 0);
  GateFluenceActorMessenger * pMessenger;

  int mCurrentEvent;
  bool mIsLastHitEventImageEnabled;
  bool mIsStepLengthImageEnabled;
  bool mIsSquaredImageEnabled;
  bool mIsUncertaintyImageEnabled;
  bool mIsNormalisationEnabled;
  bool mIsNumberOfHitsImageEnabled;
  bool mIsScatterImageEnabled;
  bool mIgnoreWeight;

  std::map<G4String, GateImage*> mProcesses;
  std::vector<G4String> mProcessName;
  std::vector<GateImage*> mFluencePerOrderImages;

  G4String mResponseFileName;
  G4String mScatterOrderFilename;
  G4String mSeparateProcessFilename;

  GateEnergyResponseFunctor mEnergyResponse;
};

MAKE_AUTO_CREATOR_ACTOR(FluenceActor, GateFluenceActor)

#endif /* end #define GATESIMULATIONSTATISTICACTOR_HH */
