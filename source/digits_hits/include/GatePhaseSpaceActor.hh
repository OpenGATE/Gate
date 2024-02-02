/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GATEPHASESPACEACTOR_HH
#define GATEPHASESPACEACTOR_HH

#include "GateVActor.hh"
#include "GateImage.hh"
#include "GateTreeFileManager.hh"

struct iaea_header_type;
struct iaea_record_type;

class G4EmCalculator;

class GatePhaseSpaceActorMessenger;

//====================================================================
class GatePhaseSpaceActor : public GateVActor
{
public:
  virtual ~GatePhaseSpaceActor();

  //====================================================================
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GatePhaseSpaceActor)

  //====================================================================
  // Constructs the sensor
  virtual void Construct();

  //====================================================================
  // Callbacks
  virtual void UserSteppingAction(const GateVVolume *, const G4Step *);

  virtual void PreUserTrackingAction(const GateVVolume *, const G4Track *);

  virtual void PostUserTrackingAction(const GateVVolume *, const G4Track *);

  virtual void BeginOfEventAction(const G4Event *e);

  virtual void BeginOfRunAction(const G4Run *r);

  virtual void RecordEndOfAcquisition();

  //=======================================================
  /// Saves the data collected to the file
  virtual void SaveData();

  virtual void ResetData();

  void InitTree();

  void SetIsXPositionEnabled(bool b) { EnableXPosition = b; }

  void SetIsYPositionEnabled(bool b) { EnableYPosition = b; }

  void SetIsZPositionEnabled(bool b) { EnableZPosition = b; }

  void SetIsEkineEnabled(bool b) { EnableEkine = b; }

  void SetIsXDirectionEnabled(bool b) { EnableXDirection = b; }

  void SetIsYDirectionEnabled(bool b) { EnableYDirection = b; }

  void SetIsZDirectionEnabled(bool b) { EnableZDirection = b; }

  void SetIsParticleNameEnabled(bool b) { EnablePartName = b; }

  void SetIsProdVolumeEnabled(bool b) { EnableProdVol = b; }

  void SetIsProdProcessEnabled(bool b) { EnableProdProcess = b; }

  void SetIsWeightEnabled(bool b) { EnableWeight = b; }

  void SetIsTimeEnabled(bool b) { EnableTime = b; }

  void SetIsIonTimeEnabled(bool b){EnableIonTime = b;}
  
  void SetIsLocalTimeEnabled(bool b) { EnableLocalTime = b; }

  void SetIsTimeFromBeginOfEventEnabled(bool b) { EnableTimeFromBeginOfEvent = b; }

  void SetTrackLengthEnabled(bool b) { EnableTrackLengthFlag = b; }

  void SetIsMassEnabled(bool b) { EnableMass = b; }

  void SetIsSecStored(bool b) { EnableSec = b; }

  void SetIsAllStep(bool b) { EnableAllStep = b; }

  void SetIsTOutEnabled(bool b) { EnableTOut = b; }

  void SetIsTProdEnabled(bool b) { EnableTProd = b; }

  void SetIsChargeEnabled(bool b) { EnableCharge = b; }

  void SetIsAtomicNumberEnabled(bool b) { EnableAtomicNumber = b; }

  void SetIsElectronicDEDXEnabled(bool b) { EnableElectronicDEDX = b; }

  void SetIsTotalDEDXEnabled(bool b) { EnableTotalDEDX = b; }

  void SetUseVolumeFrame(bool b) { mUseVolFrame = b; }

  bool GetUseVolumeFrame() { return mUseVolFrame; }

  void SetStoreOutgoingParticles(bool b) { mStoreOutPart = b; }

  bool GetStoreOutgoingParticles() { return mStoreOutPart; }

  void SetMaxFileSize(double size) { mFileSize = size; }

  double GetMaxFileSize() { return mFileSize; }

  void SetIsPrimaryEnergyEnabled(bool b) { bEnablePrimaryEnergy = b; }

  void SetIsEmissionPointEnabled(bool b) { bEnableEmissionPoint = b; }

  void SetEnableCoordFrame() { bEnableCoordFrame = true; }

  bool GetEnableCoordFrame() { return bEnableCoordFrame; }

  void SetCoordFrame(G4String nameOfFrame) { bCoordFrame = nameOfFrame; }

  G4String GetCoordFrame() { return bCoordFrame; }

  void SetIsSpotIDEnabled() { bEnableSpotID = true; }

  bool GetIsSpotIDEnabled() { return bEnableSpotID; }

  void SetSpotIDFromSource(G4String nameOfSource) { bSpotIDFromSource = nameOfSource; }

  G4String GetSpotIDFromSource() { return bSpotIDFromSource; }

  void SetEnabledCompact(bool b) { bEnableCompact = b; }

  void SetEnablePDGCode(bool b) { bEnablePDGCode = b; }

  void SetIsNuclearFlagEnabled(bool b) { EnableNuclearFlag = b; }

  void SetEnabledSphereProjection(bool b) { mSphereProjectionFlag = b; }

  void SetSphereProjectionCenter(G4ThreeVector c) { mSphereProjectionCenter = c; }

  void SetSphereProjectionRadius(double r) { mSphereProjectionRadius = r; }

  void SetEnabledTranslationAlongDirection(bool b) { mTranslateAlongDirectionFlag = b; }

  void SetTranslationAlongDirectionLength(double r) { mTranslationLength = r; }

  void SetMaskFilename(G4String filename);

  void SetKillParticleFlag(bool b);

protected:
  GatePhaseSpaceActor(G4String name, G4int depth = 0);

  G4String mFileType;
  G4int mNevent;

  GateOutputTreeFileManager *mFile;

  bool EnableCharge;
  bool EnableAtomicNumber;
  bool EnableElectronicDEDX;
  bool EnableTotalDEDX;
  bool EnableXPosition;
  bool EnableYPosition;
  bool EnableZPosition;
  bool EnableEkine;
  bool EnableXDirection;
  bool EnableYDirection;
  bool EnableZDirection;
  bool EnablePartName;
  bool EnableProdVol;
  bool EnableProdProcess;
  bool EnableWeight;
  bool EnableTime;
  bool EnableIonTime;
  bool EnableLocalTime;
  bool EnableMass;
  bool EnableSec;
  bool EnableAllStep;
  bool mUseVolFrame;
  bool mStoreOutPart;
  bool EnableNuclearFlag;
  bool EnableTimeFromBeginOfEvent;
  bool EnableTrackLengthFlag;

  bool EnableTOut;
  bool EnableTProd;

  bool mSphereProjectionFlag;
  G4ThreeVector mSphereProjectionCenter;
  double mSphereProjectionRadius;

  bool mTranslateAlongDirectionFlag;
  double mTranslationLength;

  bool bEnableCoordFrame;
  G4String bCoordFrame;
  bool bEnablePrimaryEnergy;
  float bPrimaryEnergy;
  bool bEnableEmissionPoint;
  float bEmissionPointX, bEmissionPointY, bEmissionPointZ;
  bool bEnableSpotID;
  G4String bSpotIDFromSource;
  int bSpotID;
  bool bEnableCompact;
  bool bEnablePDGCode;
  long int bPDGCode;
  double trackLength;

  bool mMaskIsEnabled;
  G4String mMaskFilename;
  GateImage mMask;
  bool mKillParticleFlag;

  bool bEnableTOut;
  bool bEnableTProd;

  double mFileSize;

  long int mNumberOfTrack;

  bool mIsFirstStep;

  char pname[256];

  G4int Za;
  G4double charge;
  float elecDEDX;
  float totalDEDX;
  float stepLength;
  float edep;

  float x;
  float y;
  float z;
  float dx;
  float dy;
  float dz;
  float e;
  float ekPost;
  float ekPre;
  float w;
  float tOut;
  float tProd;
  double t; // t is either time or local time.
  double pt; // Used for proton arrival time recovering from mTime in SourceOfPromptGamma
  double fTimeFromBeginOfEvent;
  double fBeginOfEventTime;

  G4int m;
  char vol[256];

  char creator_process[256];
  char pro_step[256];

  int trackid;
  int parentid;
  int eventid;
  int runid;

  int creator;
  int nucprocess;
  int order;

  G4EmCalculator *emcalc;
  GatePhaseSpaceActorMessenger *pMessenger;

  iaea_record_type *pIAEARecordType;
  iaea_header_type *pIAEAheader;
};

MAKE_AUTO_CREATOR_ACTOR(PhaseSpaceActor, GatePhaseSpaceActor)

#endif /* end #define GATESOURCEACTOR_HH */
