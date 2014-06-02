/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*!
  \class  GateDoseActor
  \author fabien.baldacci@creatis.insa-lyon.fr
*/

#ifndef GATETLEDOSEACTOR_HH
#define GATETLEDOSEACTOR_HH

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "GateTLEDoseActorMessenger.hh"
#include "GateImageWithStatistic.hh"
#include "GateMaterialMuHandler.hh"
#include "G4UnitsTable.hh"

class GateTLEDoseActor : public GateVImageActor
{
public:

  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateTLEDoseActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateTLEDoseActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  void EnableEdepImage(bool b) { mIsEdepImageEnabled = b; }
  void EnableDoseUncertaintyImage(bool b) { mIsDoseUncertaintyImageEnabled = b; }
  void EnableEdepSquaredImage(bool b) { mIsEdepSquaredImageEnabled = b; }
  void EnableEdepUncertaintyImage(bool b) { mIsEdepUncertaintyImageEnabled = b; }
  void EnableDoseImage(bool b) { mIsDoseImageEnabled = b; }
  void EnableDoseSquaredImage(bool b) { mIsDoseSquaredImageEnabled = b; }

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void BeginOfEventAction(const G4Event * event);

  //virtual void PostUserTrackingAction(const GateVVolume *, const G4Track* t);
  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);
  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step);
  virtual void UserPreTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}
  virtual void UserPostTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}

  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  ///Scorer related
  //virtual G4bool ProcessHits(G4Step *, G4TouchableHistory*);
  virtual void clear(){ResetData();}
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}

protected:
  GateTLEDoseActor(G4String name, G4int depth=0);
  GateTLEDoseActorMessenger * pMessenger;

  GateImageWithStatistic mDoseImage;
  //GateImageWithStatistic mPrimaryDoseImage;
  //GateImageWithStatistic mSecondaryDoseImage;
  GateImageWithStatistic mEdepImage;
  GateImage mLastHitEventImage;

  GateMaterialMuHandler* mMaterialHandler;
  G4String mDoseFilename;
  G4String mPDoseFilename;
  G4String mSDoseFilename;
  G4String mEdepFilename;
  G4double ConversionFactor;
  G4double VoxelVolume;
  bool mIsEdepImageEnabled;
  bool mIsDoseUncertaintyImageEnabled;
  bool mIsLastHitEventImageEnabled;
  bool mIsEdepSquaredImageEnabled;
  bool mIsEdepUncertaintyImageEnabled;
  bool mIsDoseImageEnabled;
  bool mIsDoseSquaredImageEnabled;
  int mCurrentEvent;
  G4double outputEnergy;
  G4double totalEnergy;
  G4double mEdep;
  G4String lastMaterial;
  G4double lastedep;
  G4int lastindex;
  G4bool firstdeposit;
  G4int primaryindex;
  std::vector<G4double> primarydeposition;
  std::vector<G4double> secondarydeposition;
};

MAKE_AUTO_CREATOR_ACTOR(TLEDoseActor,GateTLEDoseActor)

#endif /* end #define GATETLEDOSEACTOR_HH */
