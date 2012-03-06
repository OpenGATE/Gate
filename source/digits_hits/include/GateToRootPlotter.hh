/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateToRootPlotter_H
#define GateToRootPlotter_H

#include "GateConfiguration.h"

//e #ifdef G4ANALYSIS_USE_ROOT
#ifdef G4ANALYSIS_USE_ROOT_PLOTTER

#include "GateVOutputModule.hh"

#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TBranch.h"
#include "TCanvas.h"

#include "globals.hh"
#include <fstream>

#include "G4Run.hh"
#include "G4Step.hh"
#include "G4Event.hh"

#include "GateTrajectoryNavigator.hh"

class GateToRootPlotterMessenger;
class GateVVolume;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateToRootPlotter :  public GateVOutputModule
{
public:

  GateToRootPlotter(const G4String& name, GateOutputMgr* outputMgr,DigiMode digiMode);
  virtual ~GateToRootPlotter();
  const G4String& GiveNameOfFile();

  void RecordBeginOfAcquisition();
  void RecordEndOfAcquisition();
  void RecordBeginOfRun(const G4Run *);
  void RecordEndOfRun(const G4Run *);
  void RecordBeginOfEvent(const G4Event *);
  void RecordEndOfEvent(const G4Event *);
  void RecordStepWithVolume(const GateVVolume * , const G4Step *);

  //! saves the geometry voxel information
  void RecordVoxels(GateVGeometryVoxelStore *) {};

  void RecordDigitizer(const G4Event *);

  void Init();
  void Book();
  void Store();
  void Finish();
  void Reset();

  void ShowPlotter();
  void ClosePlotter();

  G4int GetRecordFlag()           { return m_recordFlag; };
  void  SetRecordFlag(G4int flag) { m_recordFlag = flag; };

  void PlotAll();
  void AddPlot(std::vector<G4String>);
  void RemovePlot(G4int number);
  void RemoveAllPlots();
  void ListPlots();
  void SetPlotWidth (G4int value);
  void SetPlotHeight(G4int value);
  void SetNColumns  (G4int value);

private:

  G4ThreeVector  m_ionDecayPos;
  G4ThreeVector  m_positronAnnihilPos;
  
  G4double m_positronKinEnergy;
  G4int    m_recordFlag;

  TFile*               m_hfile; // the file for histograms, tree ...
  
  G4int                m_updateROOTmodulo;

  TCanvas*             m_rootPlotterCanvas;

  ///// Histogrammes ///////////
  //  void Plot1D(IHistogram1D* histo);
  //  void Plot2D(IHistogram2D* histo);
  void Plot1D(G4String histoName);
  void Plot2D(G4String histoName);
  void PlotTree(G4String treeName, G4String varName, G4String cut);
  void AdjustWindow();

  typedef std::vector<G4String> GateToRootPlotterPlot;
  std::vector<GateToRootPlotterPlot*> m_plots;

  GateToRootPlotterMessenger* m_rootPlotterMessenger;

  G4int m_nPlotterColumns;
  G4int m_plotWidth;
  G4int m_plotHeight;

  G4String m_noFileName;
};

#endif
//e #endif
#endif
