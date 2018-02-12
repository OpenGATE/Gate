/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateComptonCameraActor
*/

#ifndef GATECOMPTONCAMERACTOR_HH
#define GATECOMPTONCAMERACTOR_HH

#include "GateVActor.hh"
#include "GateActorMessenger.hh"
#include "GateVolumeID.hh"
#include "GateCCRootDefs.hh"

#include "GateCrystalHit.hh"
#include "GateHitConvertor.hh"
#include "GateDigitizer.hh"
#include "GatePulseAdder.hh"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

#include "TNtuple.h"
#include "TTree.h"
#include "TBranch.h"
#include "TString.h"

class G4EmCalculator;
class GateDigitizer;

//-----------------------------------------------------------------------------
class GateComptonCameraActor : public GateVActor
{
public:

  virtual ~GateComptonCameraActor();

  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateComptonCameraActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  //-----------------------------------------------------------------------------
  // Callbacks

  virtual void BeginOfRunAction(const G4Run * r);
  virtual void BeginOfEventAction(const G4Event *) ;
  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);

  virtual void PreUserTrackingAction(const GateVVolume *, const G4Track*) ;
  virtual void PostUserTrackingAction(const GateVVolume *, const G4Track*) ;
  virtual void EndOfEventAction(const G4Event*);

  //-----------------------------------------------------------------------------
  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  virtual void Initialize(G4HCofThisEvent* HCE);
  virtual void EndOfEvent(G4HCofThisEvent*);

  int GetNDaughtersBB() {return nDaughterBB;}

  //Messenger flag
  void SetSaveHitsTreeFlag( bool b ){  mSaveHitsTreeFlag= b; }

  //! Get the digitizer
  inline GateDigitizer*   GetDigitizer()
  { return m_digitizer; }

protected:
  GateComptonCameraActor(G4String name, G4int depth=0);

  TFile * pTfile;

  GateCCHitTree*  m_hitsTree;
  GateCCRootHitBuffer  m_hitsBuffer;

  GateCCHitTree*  m_hitsAbsTree;
  GateCCRootHitBuffer  m_hitsAbsBuffer;

  GateCCHitTree*  m_hitsScatTree;
  GateCCRootHitBuffer  m_hitsScatBuffer;

  std::vector<G4String> layerNames;
  std::vector<std::unique_ptr<TTree>> pSingles;

  GateCCSingleTree*  m_SingleTree;
  GateCCRootSingleBuffer  m_SinglesBuffer;

  int slayerID;
  G4String mHistName;

  unsigned int nDaughterBB;
  G4String attachPhysVolumeName;

  double Ei,Ef;
  int nTrack;
  int nEvent;
  bool newEvt;
  bool newTrack;
  double sumNi;
  double edepTrack;
  double edepEvt;
  double tof;
  double edptempAb;

  G4String VolNameStep;
  G4int evtID;
  G4int runID;

  G4double hitEdep;
  G4ThreeVector hitPostPos;
  G4ThreeVector hitPrePos;
  G4int trackID;
  G4int  parentID;
  G4double trackLength;
  G4double trackLocalTime;
  G4ThreeVector hitPreLocalPos;

  //Test for readout output (Manual singles)
  double* edepInEachLayerEvt;

  double* xPos_InEachLayerEvt;
  double* yPos_InEachLayerEvt;
  double* zPos_InEachLayerEvt;

  GateCrystalHitsCollection* crystalCollection; //Hit collection
  static const G4String theCrystalCollectionName;//name of the hit collection
  // int m_collectionID;

  GatePulseList* crystalPulseList;
  static const G4String thedigitizerName;
  GatePulseProcessorChain* chain;

  void readPulses(GatePulseList* pPulseList);
  void processPulsesIntoSinglesTree();

  GateActorMessenger* pMessenger;

  //messenger
  bool mSaveHitsTreeFlag;

  G4EmCalculator * emcalc;
  GateDigitizer* m_digitizer;
};

MAKE_AUTO_CREATOR_ACTOR(ComptonCameraActor,GateComptonCameraActor)


#endif /* end #define GATECOMPTONCAMERAACTOR_HH */
//#endif
