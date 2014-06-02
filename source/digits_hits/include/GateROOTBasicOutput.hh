/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT
#ifndef GateROOTBasicOutput_h
#define GateROOTBasicOutput_h 1

#include "GateRecorderBase.hh"

#include "G4UserRunAction.hh"
//#include "CLHEP/Hist/Histogram.h"
//#include "CLHEP/Hist/Tuple.h"
//#include "CLHEP/Hist/HBookFile.h"
// Include files for ROOT.
#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TNtuple.h"
#include "TTree.h"
#include "TBranch.h"
#include "TRandom.h"

#include <iostream>
#include "globals.hh"
#include "G4Step.hh"
#include "G4Event.hh"
#include "G4VProcess.hh"
#include "GateUserActions.hh"

class GateVVolume;
class GateCrystalHit;
class GateROOTBasicOutputMessenger;
class G4Run;

static const int dimOfHitVector = 500 ;

class GateROOTBasicOutput: public GateRecorderBase
{
public:
  GateROOTBasicOutput();
  ~GateROOTBasicOutput();

  void RecordBeginOfRun(const G4Run *);
  void RecordEndOfRun(const G4Run *);

  void RecordBeginOfEvent(const G4Event *);
  void RecordEndOfEvent(const G4Event * );

  void RecordStepWithVolume(const GateVVolume * v, const G4Step *);
  void SetfileName(G4String name);

private:
  TFile * hfile;
  G4String fileName;

  Float_t  *Edep;
  Float_t  Etot;
  Float_t  xpos1;
  Float_t  ypos1;
  Float_t  zpos1;
  Int_t    run;

  TTree *tree;
  Int_t numberHits;

  GateROOTBasicOutputMessenger* runMessenger;
};
//---------------------------------------------------------------------------

#endif
#endif
