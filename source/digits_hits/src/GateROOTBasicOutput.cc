/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateROOTBasicOutput.hh"

#ifdef G4ANALYSIS_USE_ROOT
#include "GateROOTBasicOutputMessenger.hh"

// Include files for ROOT.
#include "Rtypes.h"
#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TNtuple.h"
#include "TTree.h"
#include "TBranch.h"
#include "TRandom.h"
#include "TObject.h"

#include "globals.hh"

#include "G4Event.hh"
#include "G4Run.hh"
#include "G4Step.hh"
#include "GateRecorderBase.hh"
#include "G4ios.hh"
#include "G4SDManager.hh"
#include "Randomize.hh"
#include "G4VVisManager.hh"
#include "G4UImanager.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "GateCrystalHit.hh"

#include "CLHEP/Random/RanecuEngine.h"

//----------------------------------------------------------------
GateROOTBasicOutput::GateROOTBasicOutput()
  : fileName("histfile")
{
  runMessenger = new GateROOTBasicOutputMessenger(this);

  static TROOT rootBase("simple","Test of histogramming and I/O");
  Edep  = new Float_t[dimOfHitVector];
}
//----------------------------------------------------------------

//----------------------------------------------------------------
GateROOTBasicOutput::~GateROOTBasicOutput()
{
  delete[] Edep;
  delete runMessenger;
}
//----------------------------------------------------------------

//----------------------------------------------------------------
void GateROOTBasicOutput::RecordBeginOfRun(const G4Run* r)
{
  run = 0;

  if ((r->GetRunID())==0){

    hfile = new TFile (fileName, "RECREATE");
    tree = new TTree("T", "Tree de Simu");
    tree->Branch("run",&run,"run/I");
    tree->Branch("numberHits",&numberHits,"numberHits/I");
    tree->Branch("Edep", Edep, "Edep[numberHits]/F");
    tree->Branch("Etot", &Etot, "Etot/F");
    tree->Branch("xpos1",&xpos1,"xpos1/F");
    tree->Branch("ypos1",&ypos1,"ypos1/F");
    tree->Branch("zpos1",&zpos1,"zpos1/F");
  }
  else
    {
      hfile = new TFile (fileName, "UPDATE");
      hfile->ls();
      tree = (TTree*)hfile->Get("T");


      TBranch* b;
      b = tree->GetBranch("run");
      b->SetAddress(&run);

      b = tree->GetBranch("numberHits");
      b->SetAddress(&numberHits);

      b = tree->GetBranch("Edep");
      b->SetAddress(Edep);

      b = tree->GetBranch("Etot");
      b->SetAddress(&Etot);

      b = tree->GetBranch("xpos1");
      b->SetAddress(&xpos1);


      b = tree->GetBranch("ypos1");
      b->SetAddress(&ypos1);

      b = tree->GetBranch("zpos1");
      b->SetAddress(&zpos1);
    }

  run = r->GetRunID();
}
//----------------------------------------------------------------

//----------------------------------------------------------------
void GateROOTBasicOutput::RecordEndOfRun(const G4Run*)
{
  hfile->Write();
  hfile->Close();
  G4cout << " The hfile  " << hfile << " is closed. " << G4endl;
}
//----------------------------------------------------------------

//----------------------------------------------------------------
void GateROOTBasicOutput::RecordBeginOfEvent(const G4Event*)
{

  for (G4int k = 0 ; k < dimOfHitVector ; k++)
    {
      Edep[k] = 0.;

    }

  xpos1 = 0.;
  ypos1 = 0.;
  zpos1 = 0.;

  Etot = 0.;
}
//-----------------------------------------------------------------

//-----------------------------------------------------------------
void GateROOTBasicOutput::RecordEndOfEvent(const G4Event* event )
{

  G4SDManager* fSDM = G4SDManager::GetSDMpointer();

  G4int collectionID = fSDM->GetCollectionID("crystalCollection");

  G4HCofThisEvent* HCofEvent = event->GetHCofThisEvent();

  GateCrystalHitsCollection* trackerHC =
    (GateCrystalHitsCollection*) (HCofEvent->GetHC(collectionID));


  numberHits = trackerHC->entries();

  //////////////////////////////////////////////////
  if ( numberHits > dimOfHitVector ) { numberHits = dimOfHitVector ; }

  GateCrystalHit* aHit;
  for (G4int i = 0; i < numberHits ; i++)
    {
      aHit = (*trackerHC)[i];

      Edep[i] = aHit->GetEdep() / keV;
      Etot += Edep[i];

      xpos1 = aHit->GetGlobalPos().x();
      ypos1 = aHit->GetGlobalPos().y();
      zpos1 = aHit->GetGlobalPos().z();
    }

  // ooOO0OOoo  Remplissage du Tree ooOO0OOoo
  tree->Fill();

  //  G4cout << " FIN GateROOTBasicOutput::RecordEndOfEvent" << G4endl;
}
//-----------------------------------------------------------------

//-----------------------------------------------------------------
void GateROOTBasicOutput::RecordStepWithVolume(const GateVVolume *, const G4Step*)
{
}
//-----------------------------------------------------------------

//-----------------------------------------------------------------
void GateROOTBasicOutput::SetfileName(G4String name)
{
  fileName = name;
  G4cout << " The ROOTBasic file name is = " << fileName << "." << G4endl;
}
//-----------------------------------------------------------------

#endif
