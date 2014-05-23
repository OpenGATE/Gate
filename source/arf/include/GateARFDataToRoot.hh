/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateARFDataToRoot_H
#define GateARFDataToRoot_H

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateVOutputModule.hh"

#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TBranch.h"

#include "globals.hh"
#include <fstream>

#include "G4Run.hh"
#include "G4Step.hh"
#include "G4Event.hh"

class GateARFDataToRootMessenger;
class GateSingleDigi;
class GateSteppingAction;
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....


class GateARFData
{ public :
      G4double m_Edep; // deposited energy
      G4double m_Y, m_X; // porjection position on the detection plane
      };

class GateARFDataToRoot :  public GateVOutputModule
{
public:

  GateARFDataToRoot(const G4String& name, GateOutputMgr* outputMgr,DigiMode digiMode);
  virtual ~GateARFDataToRoot();
  const G4String& GiveNameOfFile();

  void RecordBeginOfAcquisition();
  void RecordEndOfAcquisition();
  void RecordBeginOfRun(const G4Run *);
  void RecordEndOfRun(const G4Run *);
  void RecordBeginOfEvent(const G4Event *);
  void RecordEndOfEvent(const G4Event *);
  void RecordDigitizer(const G4Event *);
  void RecordStep(const G4Step*);
  void RecordVoxels(GateVGeometryVoxelStore*);
  void RecordStepWithVolume(const GateVVolume*, const G4Step*){}
  void RecordTracks(GateSteppingAction*){}
  void   RegisterNewSingleDigiCollection(const G4String& aCollectionName,G4bool outputFlag);
  void RegisterNewCoincidenceDigiCollection(const G4String& ,G4bool ){}
  
  void SetVerboseLevel(G4int val) 
  { 
    GateVOutputModule::SetVerboseLevel(val);
    
  }
      G4int StoreARFData(GateSingleDigi*);
      void SetProjectionPlane(G4double aX){m_Xplane = aX;}

      G4ThreeVector GetPositionAtVertex();
      void SetPositionAtVertex(G4ThreeVector);
      G4ThreeVector GetVertexMomentumDirection();
      void SetVertexMomentumDirection(G4ThreeVector);

      //! Implementation of the pure virtual method ProcessHits().
      //! This methods generates a GateCrystalHit and stores it into the SD's hit collection
      
      void CloseARFDataRootFile();

      void SetARFDataRootFileName(G4String);

      void IncrementNbOfSourcePhotons();

      long unsigned int GetNbOfGoingOutPhotons(){return NbofGoingOutPhotons;}
      long unsigned int GetNbOfInPhotons(){return  NbofGoingInPhotons;}
      void IncrementGoingInPhotons(){NbofGoingInPhotons++;}
      void IncrementGoingOutPhotons(){NbofGoingOutPhotons++;}
      void IncrementKilledInsideCrystalPhotons(){NbofKilledInsideCrystalPhotons++;}
      void IncrementKilledInsideColliPhotons(){NbofKilledInsideColliPhotons++;}
      void IncrementKilledInsideCamera(){NbofKilledInsideCamera++;}
      void IncrementInCamera(){ IN_camera++;}
      void IncrementOutCamera(){ OUT_camera++;}
      void DisplayARFStatistics();
      G4int IsCounted(){return m_iscounted;}
      G4int IsCountedOut(){return m_iscountedOut;}
      void SetCounted(){m_iscounted = 1;}
      void SetCountedOut(){m_iscountedOut = 1;}
      void SetNHeads(G4int N ){ NbOfHeads = N;}
      void setDRFDataprojectionmode( G4int opt ){ m_DRFprojectionmode = opt; }



private:

  GateARFDataToRootMessenger* m_rootMessenger;

      G4String m_ARFDatafilename; // the naeof the root output file
      TFile*  m_ARFDatafile;   // the root file
      TTree*  m_ARFDataTree; // the root tree 
      TTree*  m_NbOfPhotonsTree;

      // the datas to be saved in the root file

      GateARFData  theData;

      G4int m_DRFprojectionmode;


      G4String m_SingleDigiCollectionName; // the singledigi collection name


      G4RotationMatrix m_theRotation;
      G4ThreeVector m_theTranslation;
      G4double m_Xplane; // this is the YZ projection plane where we project energy deposition coordinates
      ULong64_t IN_camera;
      ULong64_t OUT_camera;
      ULong64_t NbOfSimuPhotons;
      ULong64_t NbOfSourcePhotons;
      ULong64_t NbofGoingOutPhotons;
      ULong64_t NbofStraightPhotons;
      ULong64_t NbofGoingInPhotons;
      ULong64_t NbofKilledInsideCrystalPhotons;
      ULong64_t NbofKilledInsideColliPhotons;
      ULong64_t NbofKilledInsideCamera;
      ULong64_t NbofBornInsidePhotons;
      ULong64_t NbofStoredPhotons;
      G4int m_GoingIn, m_GoingOut; // number of times a photons enters the ARFSimu SD
      G4int m_iscounted, m_iscountedOut; // flag to know the in going photon has been counted or not
      G4int headID;
      G4int NbOfHeads;
};

#endif
#endif
