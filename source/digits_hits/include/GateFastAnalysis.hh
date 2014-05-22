/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#ifndef GateFastAnalysis_H
#define GateFastAnalysis_H

#include "GateVOutputModule.hh"
#include "GateDetectorConstruction.hh"

class GateFastAnalysisMessenger;
class GateVVolume;

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

};

#endif
#endif
