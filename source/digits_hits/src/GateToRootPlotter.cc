/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateToRootPlotter.hh"

#ifdef G4ANALYSIS_USE_ROOT
#ifdef G4ANALYSIS_USE_ROOT_PLOTTER

#include "GateToRootPlotterMessenger.hh"

#include "globals.hh"

#include "G4Run.hh"
#include "G4Step.hh"
#include "G4Event.hh"
#include "GateCrystalHit.hh"
#include "GatePhantomHit.hh"
#include "G4VHitsCollection.hh"

#include "G4VProcess.hh"
#include "GateRecorderBase.hh"
#include "G4ios.hh"
//LF
//#include <iomanip.h>
#include <iomanip>
//LF
#include "G4UImanager.hh"
#include "GatePrimaryGeneratorAction.hh"

#include "G4ParticleDefinition.hh"
#include "G4Positron.hh"
#include "G4GenericIon.hh"

#include "GateDetectorConstruction.hh"
#include "GateVVolume.hh"

#include "GateDigitizer.hh"
#include "GateSingleDigi.hh"
#include "GateCoincidenceDigi.hh"

#include "GateSourceMgr.hh"
#include "GateOutputMgr.hh"

#include "TROOT.h"
//#include "TApplication.h"
//#include "TGClient.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateToRootPlotter::GateToRootPlotter(const G4String& name, GateOutputMgr* outputMgr,DigiMode digiMode)
  : GateVOutputModule(name,outputMgr,digiMode)
  , m_updateROOTmodulo(10)
{

  m_isEnabled = true; // This module is always enabled but only use when
                      // compiling with the G4ANALYSIS_USE_ROOT_PLOTTER
  nVerboseLevel = 0;
  m_rootPlotterMessenger = new GateToRootPlotterMessenger(this);

  Init();
  //! We book histos and ntuples only once per application
  Book();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo...

GateToRootPlotter::~GateToRootPlotter() 
{
  Finish();
  delete m_rootPlotterMessenger;
  if (nVerboseLevel > 0) G4cout << "GateToRootPlotter deleting..." << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

const G4String& GateToRootPlotter::GiveNameOfFile()
{
  m_noFileName = "  "; // 2 spaces for output module with no fileName
  return m_noFileName;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::Init()
{
  if (nVerboseLevel > 2)
    G4cout << "GateToRootPlotter::Init" << G4endl;

  nVerboseLevel = 0;

  if (nVerboseLevel > 0) G4cout << "GateToRootPlotter: ROOT: files creation..." << G4endl;

  //  TROOT simple("simple","simple histogramming");
  //  m_hfile = new TFile("gate.root","READ");

  //  int argc = 1; 
  //  char **argv;
  //  TApplication theApp("App", &argc, argv);
  //  TApplication *theApp = new TApplication("App", ((int*)0), ((char**)0));

  m_nPlotterColumns = 1;
  m_plotWidth       = 300;
  m_plotHeight      = 300;

  //  ShowPlotter();

}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::ShowPlotter()
{
  if (((TCanvas*)gROOT->FindObject("pw")) == NULL) {
    //  if (!m_rootPlotterCanvas) {
    m_rootPlotterCanvas  = new TCanvas("pw", "Gate Plotter", 600, 600);
    gSystem->ProcessEvents();
    //  theApp.Run(kTRUE);

    AdjustWindow();
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::ClosePlotter()
{
  if (((TCanvas*)gROOT->FindObject("pw")) != NULL) {
    m_rootPlotterCanvas->Close();
    //    delete m_rootPlotterCanvas;
    gSystem->ProcessEvents();
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::Book()
{
  if (nVerboseLevel > 2)
    G4cout << "GateToRootPlotter::Book" << G4endl;

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo...

void GateToRootPlotter::Finish() 
{
  if (nVerboseLevel > 2)
    G4cout << "GateToRootPlotter::Finish" << G4endl;

  //! We close the histos only at the end of the application
  PlotAll();
  Store();

  if (nVerboseLevel > 0) G4cout << "GateToRootPlotter: ROOT: files writing..." << G4endl;
  //  m_hfile->Write();
  if (nVerboseLevel > 0) G4cout << "GateToRootPlotter: ROOT: files closing..." << G4endl;
  //  m_hfile->Close();

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....


void GateToRootPlotter::RecordBeginOfAcquisition()
{
  if (nVerboseLevel > 2)
    G4cout << "GateToRootPlotter::RecordBeginOfAcquisition" << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....


void GateToRootPlotter::RecordEndOfAcquisition()
{
  if (nVerboseLevel > 2)
    G4cout << "GateToRootPlotter::RecordEndOfAcquisition" << G4endl;
}



//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::RecordBeginOfRun(const G4Run* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateToRootPlotter::RecordBeginOfRun" << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::RecordEndOfRun(const G4Run* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateToRootPlotter::RecordEndOfRun" << G4endl;

  PlotAll();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::RecordBeginOfEvent(const G4Event* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateToRootPlotter::RecordBeginOfEvent" << G4endl;
  //! if the flag is not set to >0, don't record the event
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::RecordEndOfEvent(const G4Event* event)
{
  if (nVerboseLevel > 2)
    G4cout << "GateToRootPlotter::RecordEndOfEvent" << G4endl;

  G4TrajectoryContainer * trajectoryContainer = event->GetTrajectoryContainer();

  if (!trajectoryContainer) {
    if (nVerboseLevel > 0) G4cout 
      << "GateToRootPlotter::RecordEndOfEvent : WARNING : G4TrajectoryContainer not found" << G4endl;
  } else {

    GateCrystalHitsCollection* CHC = GetOutputMgr()->GetCrystalHitCollection();
    if (CHC) {
      if ((event->GetEventID())%m_updateROOTmodulo == 0) {
	gSystem->ProcessEvents();
      }
    }    

    RecordDigitizer(event);

  }

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::RecordDigitizer(const G4Event* ) 
{
  if (nVerboseLevel > 2)
    G4cout << "GateToRootPlotter::RecordDigitizer" << G4endl;  
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::RecordStep(const GateVVolume * v, const G4Step* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateToRootPlotter::RecordStep" << G4endl;

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::Reset()
{

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::Store()
{
  // store histograms

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....


void GateToRootPlotter::AdjustWindow()
{
  if (((TCanvas*)gROOT->FindObject("pw")) == NULL) return;
  //  if (!m_rootPlotterCanvas) return;

  if (nVerboseLevel > 2) 
    G4cout << "GateToRootPlotter::AdjustWindow" << G4endl;
  G4int nPlots = m_plots.size();
  if (nPlots == 0) nPlots = 1;

  G4int nRows = (nPlots-1)/m_nPlotterColumns + 1;

  UInt_t ww = m_plotWidth  * m_nPlotterColumns;
  UInt_t wh = m_plotHeight * nRows;
  m_rootPlotterCanvas->SetWindowSize(ww,wh);
  m_rootPlotterCanvas->Resize();
  m_rootPlotterCanvas->cd();
  m_rootPlotterCanvas->Clear();
  m_rootPlotterCanvas->Divide(m_nPlotterColumns,nRows);

  PlotAll();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....


void GateToRootPlotter::PlotAll()
{
  if (nVerboseLevel > 1) G4cout << "GateToRootPlotter: ROOT: Plotting the histos..." << G4endl;

  if (((TCanvas*)gROOT->FindObject("pw")) == NULL) return;
  //  if (!m_rootPlotterCanvas) return;

  for (unsigned int iPlot = 0; iPlot<m_plots.size(); iPlot++) {
    m_rootPlotterCanvas->cd(iPlot+1);
    
    if ((*(m_plots[iPlot]))[0] == G4String("hist")) {
      // Note: unprotected for 2D plots: to add something like: if *(m_plots[iPlot]))[1].find(":") ...
      Plot1D((*(m_plots[iPlot]))[1]);
    } else if ((*(m_plots[iPlot]))[0] == G4String("tree")) {
      if ( (*(m_plots[iPlot])).size() == 3) {
	PlotTree((*(m_plots[iPlot]))[1], (*(m_plots[iPlot]))[2],"");
      } else if ( (*(m_plots[iPlot])).size() == 4) {
	PlotTree((*(m_plots[iPlot]))[1], (*(m_plots[iPlot]))[2], (*(m_plots[iPlot]))[3]);
      }
    }
    m_rootPlotterCanvas->Update();
  }
  gSystem->ProcessEvents();

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::Plot1D(G4String histoName)
{
  if (((TCanvas*)gROOT->FindObject("pw")) == NULL) return;

  TH1F* hist=NULL;
  if ((hist=(TH1F*)gROOT->FindObject(histoName))!=NULL) {
    hist->Draw();
  } else {
    if (nVerboseLevel > 0) G4cout << "GateToRootPlotter:  ROOT: Cannot find histo"
				   << histoName << G4endl;
  }

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::Plot2D(G4String histoName)
{
  if (((TCanvas*)gROOT->FindObject("pw")) == NULL) return;

  TH2F* hist = NULL;
  if ((hist=(TH2F*)gROOT->FindObject(histoName))!=NULL) {
    hist->Draw();
  } else {
    if (nVerboseLevel > 0) G4cout << "GateToRootPlotter:  ROOT: Cannot find histo"
				   << histoName << G4endl;
  }

}

void GateToRootPlotter::PlotTree(G4String treeName, G4String varName, G4String cut)
{
  if (((TCanvas*)gROOT->FindObject("pw")) == NULL) return;

  TTree* tree=NULL;
  if ((tree=(TTree*)gROOT->FindObject(treeName))!=NULL) {
    tree->Draw(varName.c_str(),cut);
  } else {
    if (nVerboseLevel > 0) G4cout << "GateToRootPlotter:  ROOT: Cannot find tree"
				   << treeName << G4endl;
  }

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::AddPlot(std::vector<G4String> param)
{
  if (param.size() < 1) {
    G4cout << "GateToRootPlotter::AddPlot: missing parameter(s)" << G4endl;
    return;
  } else {
    if (param[0] == G4String("hist")) {
      if (param.size() < 2) {
	G4cout << "GateToRootPlotter::AddPlot: missing parameter(s)" << G4endl;
	return;
      }
    } else if (param[0] == G4String("tree")) {
      if (param.size() < 3) {
	G4cout << "GateToRootPlotter::AddPlot: missing parameter(s)" << G4endl;
	return;
      }      
    } else {
      G4cout << "GateToRootPlotter::AddPlot: wrong plot type" << G4endl;
      G4cout << "GateToRootPlotter::AddPlot: possible choices: hist/tree" << G4endl;
      return;
    }
    GateToRootPlotterPlot* plot = new GateToRootPlotterPlot(param);
    m_plots.push_back(plot);
    AdjustWindow();
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::RemovePlot(G4int number)
{
  if (number < 0 || number > (int)(m_plots.size()-1)) {
    G4cout << "GateToRootPlotter::RemovePlot: non existing plot" << G4endl;
  } else {
    std::vector<GateToRootPlotterPlot*>::iterator plotItr;
    plotItr = m_plots.begin();
    plotItr = plotItr + number;
    delete *plotItr;
    m_plots.erase(plotItr);
  }

  AdjustWindow();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
void GateToRootPlotter::RemoveAllPlots()
{
  for (unsigned int iPlot = 0; iPlot<m_plots.size(); iPlot++) {
    RemovePlot(iPlot);
  }

  AdjustWindow();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToRootPlotter::ListPlots()
{
  G4cout << "List of inserted plots: " << G4endl;
  for (unsigned int iPlot = 0; iPlot<m_plots.size(); iPlot++) {
    G4cout << "Plot # " << iPlot;
    for (unsigned int iStr = 0; iStr<m_plots[iPlot]->size(); iStr++) {
      G4cout << " " << (*(m_plots[iPlot]))[iStr];
    }
    G4cout << G4endl; 
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
void GateToRootPlotter::SetPlotWidth (G4int value)
{ 
  m_plotWidth         = value; 
  AdjustWindow(); 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
void GateToRootPlotter::SetPlotHeight(G4int value) 
{ 
  m_plotHeight        = value; 
  AdjustWindow(); 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
void GateToRootPlotter::SetNColumns  (G4int value)
{ 
  m_nPlotterColumns   = value; 
  AdjustWindow(); 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#endif
#endif






