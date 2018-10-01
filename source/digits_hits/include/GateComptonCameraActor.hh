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
#include "G4EventManager.hh"

#include "GateCrystalHit.hh"
#include "GateHitConvertor.hh"
#include "GateDigitizer.hh"
#include "GatePulseAdder.hh"
#include "GateCCCoincidenceDigi.hh"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include  <iomanip>

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

  virtual void EndOfEventAction(const G4Event*);

  //-----------------------------------------------------------------------------
  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  virtual void Initialize(G4HCofThisEvent* ){};
  virtual void EndOfEvent(G4HCofThisEvent*){};

  int GetNDaughtersBB() {return nDaughterBB;}

  //Messenger flags
  void SetSaveHitsTreeFlag( bool b ){  mSaveHitsTreeFlag= b; }
  void SetSaveSinglesTreeFlag( bool b ){  mSaveSinglesTreeFlag= b; }
  void SetSaveCoincidencesTreeFlag( bool b ){  mSaveCoincidencesTreeFlag= b; }
  void SetSaveCoincidenceChainsTreeFlag( bool b ){  mSaveCoincidenceChainsTreeFlag= b; }

  void SetSaveHitsTextFlag( bool b ){  mSaveHitsTextFlag= b; }
  void SetSaveSinglesTextFlag( bool b ){  mSaveSinglesTextFlag= b; }
  void SetSaveCoincidenceTextFlag( bool b ){  mSaveCoincTextFlag= b; }
  void SetSaveCoincidenceChainsTextFlag( bool b ){  mSaveCoinChainsTextFlag= b; }

  void SetNumberOfDiffScattererLayers( int numS){mNumberDiffScattLayers=numS;}
  void SetNumberOfTotScattererLayers( int numS){mNumberTotScattLayers=numS;}
  void SetNameOfScattererSDVol(G4String name){mNameOfScattererSDVol=name;}
  void SetNameOfAbsorberSDVol(G4String name ) {mNameOfAbsorberSDVol=name;}

  //! Get the digitizer
  inline GateDigitizer*   GetDigitizer()
  { return m_digitizer; }

protected:
  GateComptonCameraActor(G4String name, G4int depth=0);



  void OpenTextFile(G4String initial_filename, G4String specificName, std::ofstream & oss);
 void OpenTextFile(G4String initial_filename, std::vector<G4String> specificN, std::vector<std::shared_ptr<std::ofstream> > &ss);

  void SaveAsTextHitsEvt(GateCrystalHit* aHit, std::string layerN);
  void SaveAsTextSingleEvt(GateSingleDigi *aSin);
  void SaveAsTextCoincEvt(GateCCCoincidenceDigi* aCoin,std::ofstream& ossC);
  void closeTextFiles();

  std::vector<G4String> layerNames;
  int slayerID;
  G4String mHistName;

  TFile * pTfile;

  GateCCHitTree*  m_hitsTree;
  GateCCRootHitBuffer  m_hitsBuffer;

  GateCCSingleTree*  m_SingleTree;
  GateCCRootSingleBuffer  m_SinglesBuffer;

  GateCCRootCoincBuffer  m_CoincBuffer;
  GateCCCoincTree*  m_CoincTree;


  std::vector<std::unique_ptr<GateCCCoincTree>> m_coincChainTree;
   // I think thath I can use the sam ebuffer maybe ?
    //std::vector<GateCCRootCoincBuffer> m_coincChainBuffer;




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

  G4String processPostStep;
  G4String VolNameStep;
  G4int evtID;
  G4int runID;

  G4double hitEdep;
  G4double Ef_oldPrimary;
  G4ThreeVector hitPostPos;
  G4ThreeVector hitPrePos;
  G4ThreeVector sourcePos;
  G4double sourceEkine;
  G4int trackID;
  G4int  parentID;
  G4double trackLength;
  G4double trackLocalTime;
  G4ThreeVector hitPreLocalPos;
   G4ThreeVector hitPostLocalPos;


   //Vector fo the hit collection since GateCrystalHistsCollection is not freeing memeory easily
   std::vector<GateCrystalHit*> hitsList;




 // GatePulseList* crystalPulseList;
   static const G4String thedigitizerName;
     GatePulseProcessorChain* chain;
    static const G4String thedigitizerSorterName;
     GateCoincidenceSorter* coincidenceSorter;

     void readPulses(GatePulseList* pPulseList);
     void processPulsesIntoSinglesTree();


  //messenger
  bool mSaveHitsTreeFlag;
  bool mSaveSinglesTreeFlag;
  bool mSaveCoincidencesTreeFlag;
  bool mSaveCoincidenceChainsTreeFlag;

  bool mSaveHitsTextFlag;
  bool mSaveSinglesTextFlag;
  bool mSaveCoincTextFlag;
  bool mSaveCoinChainsTextFlag;

  int mNumberDiffScattLayers;
   int mNumberTotScattLayers;
  G4String  mNameOfScattererSDVol;
  G4String  mNameOfAbsorberSDVol;



  std::ofstream ossHits;
  std::ofstream ossSingles;
  std::ofstream ossCoincidences;
  std::vector<std::shared_ptr<std::ofstream> > ossCoincidenceChains;
  std::vector<G4String> coincidenceChainNames;




 GateActorMessenger* pMessenger;
  G4EmCalculator * emcalc;
  GateDigitizer* m_digitizer;

};

MAKE_AUTO_CREATOR_ACTOR(ComptonCameraActor,GateComptonCameraActor)


#endif /* end #define GATECOMPTONCAMERAACTOR_HH */
//#endif
