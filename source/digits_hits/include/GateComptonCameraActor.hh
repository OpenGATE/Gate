/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
//Ane Etxebeste 01/2020
/*!
  \class  GateComptonCameraActor
*/

#ifndef GATECOMPTONCAMERACTOR_HH
#define GATECOMPTONCAMERACTOR_HH

#include "GateVActor.hh"
#include "GateActorMessenger.hh"
#include "GateVolumeID.hh"
#include "G4EventManager.hh"
#include "GateHit.hh"
#include "GateHitConvertor.hh"
#include "GateDigitizer.hh"
#include "GateCCCoincidenceDigi.hh"
#include "GatePrimTrackInformation.hh"

#include "GateCCRootDefs.hh"
#include "GateTreeFileManager.hh"

#include  <iomanip>


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

  virtual void PreUserTrackingAction(const GateVVolume *, const G4Track*);
  //test nov
  virtual void PostUserTrackingAction(const GateVVolume *, const G4Track*);
  virtual void EndOfEventAction(const G4Event*);

  //-----------------------------------------------------------------------------
  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();


  virtual void Initialize(G4HCofThisEvent* ){}
  virtual void EndOfEvent(G4HCofThisEvent*){}

  unsigned int GetNDaughtersBB() {return nDaughterBB;}

  //Messenger flags
  void SetSaveHitsTreeFlag( bool b ){  mSaveHitsTreeFlag= b; }
  void SetSaveSinglesTreeFlag( bool b ){  mSaveSinglesTreeFlag= b; }
  void SetSaveCoincidencesTreeFlag( bool b ){  mSaveCoincidencesTreeFlag= b; }
  void SetSaveCoincidenceChainsTreeFlag( bool b ){  mSaveCoincidenceChainsTreeFlag= b; }
  void SetSaveEventInfoTreeFlag( bool b ){  mSaveEventInfoTreeFlag= b; }

  void SetNumberOfDiffScattererLayers( int numS){mNumberDiffScattLayers=numS;}
  void SetNumberOfTotScattererLayers( int numS){mNumberTotScattLayers=numS;}
  void SetNameOfScattererSDVol(G4String name){mNameOfScattererSDVol=name;}
  void SetNameOfAbsorberSDVol(G4String name ) {mNameOfAbsorberSDVol=name;}

  void SetParentIDSpecificationFlag( bool b ){  mParentIDSpecificationFlag= b; }
  void SetParentIDFileName(G4String name ) {mParentIDFileName=name;}


  //Information stored in the tree handle by messenger flags
  void SetIsEnergyEnabled(bool b){EnableEnergy = b;}
  void SetIsEnergyIniEnabled(bool b){EnableEnergyIni = b;}
  void SetIsEnergyFinEnabled(bool b){EnableEnergyFin = b;}
  void SetIsTimeEnabled(bool b){EnableTime = b;}
  void SetIsXPositionEnabled(bool b){EnableXPosition = b;}
  void SetIsYPositionEnabled(bool b){EnableYPosition = b;}
  void SetIsZPositionEnabled(bool b){EnableZPosition = b;}
  void SetIsXLocalPositionEnabled(bool b){EnableXLocalPosition = b;}
  void SetIsYLocalPositionEnabled(bool b){EnableYLocalPosition = b;}
  void SetIsZLocalPositionEnabled(bool b){EnableZLocalPosition = b;}
  void SetIsXSourcePositionEnabled(bool b){EnableXSourcePosition = b;}
  void SetIsYSourcePositionEnabled(bool b){EnableYSourcePosition = b;}
  void SetIsZSourcePositionEnabled(bool b){EnableZSourcePosition = b;}
  void SetIsVolumeIDEnabled(bool b){EnableVolumeID=b;}
  void SetIsSourceEnergyEnabled(bool b){EnableSourceEnergy=b;}
  void SetIsSourcePDGEnabled(bool b){EnableSourcePDG=b;}
  void SetIsnCrystalComptEnabled(bool b){EnablenCrystalCompt=b;}
  void SetIsnCrystalRaylEnabled(bool b){EnablenCrystalRayl=b;}
  void SetIsnCrystalConvEnabled(bool b){EnablenCrystalConv=b;}

  //! Get the digitizer
  inline GateDigitizer*   GetDigitizer()
  { return m_digitizer; }

protected:
  GateComptonCameraActor(G4String name, G4int depth=0);
  std::vector<G4String> layerNames;
  int slayerID;
  G4String mHistName;



  //std::vector<std::shared_ptr<std::ofstream> > ossCoincidenceChains;
  std::vector<G4String> coincidenceChainNames;

  //Using new GateFileManager classes 12/2019
  GateOutputTreeFileManager mFileHits;
  GateOutputTreeFileManager mFileSingles;
  GateOutputTreeFileManager mFileCoinc;
  GateOutputTreeFileManager mFileEvent;
  std::vector<std::unique_ptr<GateOutputTreeFileManager>> mVectorFileCoinChain;

  GateCCRootHitBuffer  m_HitsBuffer;
  GateCCRootSingleBuffer  m_SinglesBuffer;
  GateCCRootCoincBuffer  m_CoincBuffer;
//==============================
  //Inmformation saved in the files
  bool EnableEnergy;
  bool EnableEnergyIni;
  bool EnableEnergyFin;
  bool EnableTime;
  bool EnableXPosition;
  bool EnableYPosition;
  bool EnableZPosition;
  bool EnableXLocalPosition;
  bool EnableYLocalPosition;
  bool EnableZLocalPosition;
  bool EnableXSourcePosition;
  bool EnableYSourcePosition;
  bool EnableZSourcePosition;
  bool EnableVolumeID;
  bool EnableSourceEnergy;
  bool EnableSourcePDG;
  bool EnablenCrystalCompt;
  bool EnablenCrystalConv;
  bool EnablenCrystalRayl;

 //========================
  unsigned int nDaughterBB;
  G4String attachPhysVolumeName;

  double Ei,Ef;


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

  G4int nElectronEscapedEvt;
  G4double energyElectronEscapedEvt;
  G4bool IseExitingSDVol;
  G4String eEspVolName;


  G4int nCrystalConv;
  G4int nCrystalCompt;
  G4int nCrystalRayl;

  //std::vector<double> nCrystalCompt_posZ;
  //std::vector<double> nCrystalConv_posZ;
  std::vector<double> nCrystalCompt_gTime;
  std::vector<double> nCrystalConv_gTime;
  std::vector<double> nCrystalRayl_gTime;


  G4double hitEdep;
  G4double Ef_oldPrimary;
  G4ThreeVector hitPostPos;
  G4ThreeVector hitPrePos;
  G4ThreeVector sourcePos;
  G4double sourceEnergy;
  G4int sourcePDG;
  G4int trackID;
  G4int  parentID;
  G4double trackLength;
  G4double trackLocalTime;
  G4ThreeVector hitPreLocalPos;
  G4ThreeVector hitPostLocalPos;


  //Vector fo the hit collection since GateCrystalHistsCollection is not freeing memeory easily
  std::vector<GateHit*> hitsList;




  // GatePulseList* crystalPulseList;
  static const G4String thedigitizerName;
  GatePulseProcessorChain* chain;
  static const G4String thedigitizerSorterName;
  GateCoincidenceSorterOld* coincidenceSorter;

  void readPulses(GatePulseList* pPulseList);
  void processPulsesIntoSinglesTree();


  //messenger
  bool mSaveHitsTreeFlag;
  bool mSaveSinglesTreeFlag;
  bool mSaveCoincidencesTreeFlag;
  bool mSaveCoincidenceChainsTreeFlag;
  bool mSaveEventInfoTreeFlag;

  int mNumberDiffScattLayers;
  int mNumberTotScattLayers;
  G4String  mNameOfScattererSDVol;
  G4String  mNameOfAbsorberSDVol;

  bool mParentIDSpecificationFlag;
  G4String  mParentIDFileName;
  std::vector<G4int> specfParentID;
  std::vector<G4int>::iterator itPrtID;




  //GatePrimTrackInformation* trackInfo;

  GateActorMessenger* pMessenger;
  G4EmCalculator * emcalc;
  GateDigitizer* m_digitizer;

};

MAKE_AUTO_CREATOR_ACTOR(ComptonCameraActor,GateComptonCameraActor)


#endif /* end #define GATECOMPTONCAMERAACTOR_HH */
//#endif
