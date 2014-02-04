/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

#ifndef GATEPHASESPACEACTOR_HH
#define GATEPHASESPACEACTOR_HH

#include "TROOT.h"
#include "TFile.h"
#include "TNtuple.h"
#include "TTree.h"
#include "TBranch.h"
#include "TString.h"

#include "GateVActor.hh"
#include "GatePhaseSpaceActorMessenger.hh"

struct iaea_header_type;
struct iaea_record_type;

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
  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);
  virtual void PreUserTrackingAction(const GateVVolume *, const G4Track*);
  //virtual void BeginOfEventAction(const G4Event * e);

  //=======================================================
  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  void SetIsXPositionEnabled(bool b){EnableXPosition = b;}
  void SetIsYPositionEnabled(bool b){EnableYPosition = b;}
  void SetIsZPositionEnabled(bool b){EnableZPosition = b;}
  void SetIsEkineEnabled(bool b){EnableEkine = b;}
  void SetIsXDirectionEnabled(bool b){EnableXDirection = b;}
  void SetIsYDirectionEnabled(bool b){EnableYDirection = b;}
  void SetIsZDirectionEnabled(bool b){EnableZDirection = b;}
  void SetIsParticleNameEnabled(bool b){EnablePartName = b;}
  void SetIsProdVolumeEnabled(bool b){EnableProdVol = b;}
  void SetIsProdProcessEnabled(bool b){EnableProdProcess = b;}
  void SetIsWeightEnabled(bool b){EnableWeight = b;}
  void SetIsTimeEnabled(bool b){EnableTime = b;}
  void SetIsMassEnabled(bool b){EnableMass = b;}
  void SetIsSecStored(bool b){EnableSec = b;}
  void SetIsAllStep(bool b){EnableAllStep = b;}

  void SetUseVolumeFrame(bool b){mUseVolFrame=b;}
  bool GetUseVolumeFrame(){return mUseVolFrame;}

  void SetStoreOutgoingParticles(bool b){mStoreOutPart=b;}
  bool GetStoreOutgoingParticles(){return mStoreOutPart;}

  void SetMaxFileSize(double size){mFileSize=size;}
  double GetMaxFileSize(){return mFileSize ;}


protected:
  GatePhaseSpaceActor(G4String name, G4int depth=0);

  TString mFileType;
  G4int mNevent;

  TFile * pFile;
  TTree * pListeVar;

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
  bool EnableMass;
  bool EnableSec;
  bool EnableAllStep;
  bool mUseVolFrame;
  bool mStoreOutPart;

  double mFileSize;

  long int mNumberOfTrack;

  bool mIsFistStep;

  Char_t  pname[256];
  float x;
  float y;
  float z;
  float dx;
  float dy;
  float dz;
  float e;
  float w;
  float t;
  float m;
  Char_t vol[256];

  Char_t pro_track[256];
  Char_t pro_step[256];

  int trackid;
  int eventid;
  int runid;

  GatePhaseSpaceActorMessenger* pMessenger;

  iaea_record_type *pIAEARecordType;
  iaea_header_type *pIAEAheader;
};

MAKE_AUTO_CREATOR_ACTOR(PhaseSpaceActor,GatePhaseSpaceActor)


#endif /* end #define GATESOURCEACTOR_HH */
#endif
