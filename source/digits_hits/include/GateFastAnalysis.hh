/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*  Update: V. Cuplov   15 Feb. 2012
            New output file (ntuple) dedicated to the Optical Photon Validation. 
*/

#ifdef GATE_USE_OPTICAL

#ifndef GateFastAnalysis_H
#define GateFastAnalysis_H

#include "GateVOutputModule.hh"
#include "GateDetectorConstruction.hh"

class GateFastAnalysisMessenger; 
class GateVVolume;

// v. cuplov 15.02.12
class GateTrajectoryNavigator; 
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
// v. cuplov 15.02.12

//! Faster alternative for GateAnalysis class
/**
  * GateFastAnalysis does the same as GateAnalysis except reconstructing the
  * event tree, which is very slow for large numbers of events. Some
  * information is therefore missing from the resulting pulses. The 
  * information that is missing is:
  * - source position (set to -1)
  * - NPhantomCompton (set to -1)
  * - NPhantomRayleigh (set to -1)
  * - ComptonVolumeName (set to "NULL")
  * - RayleighVolumeName (set to "NULL")
  * - PhotonID (set to -1)
  * - PrimaryID (set to -1)
  * - NCrystalCompton (set to -1)
  * - NCrystalRayleigh (set to -1)
  *
  * In order to use GateFastAnalysis the user should do
  * /gate/output/analysis disable
  * /gate/output/fastanalysis enable
  * Enabling both modules at the same time does not infleunce the
  * results, but might result in slightly longer simulation times.
  * */
class GateFastAnalysis :  public GateVOutputModule
{
public:

  GateFastAnalysis(const G4String& name, GateOutputMgr* outputMgr, DigiMode digiMode);
  virtual ~GateFastAnalysis();
  const G4String& GiveNameOfFile();

  void RecordBeginOfAcquisition();
  void RecordEndOfAcquisition();
  void RecordBeginOfRun(const G4Run * );
  void RecordEndOfRun(const G4Run * );
  void RecordBeginOfEvent(const G4Event * );
  void RecordEndOfEvent(const G4Event * );
  void RecordStepWithVolume(const GateVVolume * , const G4Step * );
  void RecordVoxels(GateVGeometryVoxelStore *) {};

  virtual void SetVerboseLevel(G4int val);

private:

  GateFastAnalysisMessenger* m_messenger;
  G4String m_noFileName;

// v. cuplov 15.02.12
        G4int nPhantomOpticalRayleigh;
        G4int nPhantomOpticalMie;
        G4int nPhantomOpticalAbsorption;
        G4int nPhantomTransport;
        G4int nCrystalOpticalRayleigh;
        G4int nCrystalOpticalMie;
        G4int nCrystalOpticalAbsorption;
        G4int nCrystalTransport;

        G4double CrystalOpticalAbsorption_x,CrystalOpticalAbsorption_y,CrystalOpticalAbsorption_z;
        G4double PhantomOpticalAbsorption_x,PhantomOpticalAbsorption_y,PhantomOpticalAbsorption_z;
        G4double CrystalOpticalPhoton_x,CrystalOpticalPhoton_y,CrystalOpticalPhoton_z;
        G4double PhantomOpticalPhoton_x,PhantomOpticalPhoton_y,PhantomOpticalPhoton_z;

        TFile*  m_opticalfile; // the file for histograms, tree ...
        GateTrajectoryNavigator* m_trajectoryNavigator;
        TTree *OpticalTuple; // new ntuple
// v. cuplov 15.02.12

};

#endif
#endif
