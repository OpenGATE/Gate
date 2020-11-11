/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GateExtendedAnalysis_H
#define GateExtendedAnalysis_H

#include "GateVOutputModule.hh"
#include "G4ThreeVector.hh"
#include "GateCrystalHit.hh"
#include "GateExtendedAnalysisMessenger.hh"

#include <memory>
#include <map>

class GateTrajectoryNavigator;
class GateVVolume;

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: Version of GateAnalysis for gammas emitted from ExtendedVSource. 
 *  Provides correct statistics about interactions (compton and rayleigh scatterings) for these gammas.
 **/
class GateExtendedAnalysis :  public GateVOutputModule
{
 public:

  GateExtendedAnalysis(const G4String& name, GateOutputMgr* outputMgr,DigiMode digiMode);
  virtual ~GateExtendedAnalysis() = default;
  
  virtual const G4String& GiveNameOfFile() override;
  virtual void RecordBeginOfAcquisition() override;
  virtual void RecordEndOfAcquisition() override;
  virtual void RecordBeginOfRun(const G4Run * ) override;
  virtual void RecordEndOfRun(const G4Run * ) override;
  virtual void RecordBeginOfEvent(const G4Event * ) override;
  virtual void RecordEndOfEvent(const G4Event * ) override;
  virtual void RecordStepWithVolume(const GateVVolume * v, const G4Step * ) override;
  virtual void RecordVoxels(GateVGeometryVoxelStore *) override;
  
  struct InteractionsInfo
  {
   G4int n_phantom_compton = 0;
   G4int n_phantom_rayleigh = 0;
   
   G4int n_crystal_compton = 0;
   G4int n_crystal_rayleigh = 0;
   
   G4String compton_volume_name = "NULL";
   G4String rayleigh_volume_name = "NULL";
  };
  
 private:
  std::map<G4int,InteractionsInfo> GetPositroniumGammasID( GateCrystalHitsCollection* chc );
  void FillPhantomInteractions( std::map<G4int,InteractionsInfo>& tracks_id );
  G4bool IsPositroniumGamma( const G4int track_id, const std::map<G4int,InteractionsInfo>& tracks_id );
  G4String GetProcessVolumeName( const G4ThreeVector& hit_pos );
  bool EventHasSources();
  void UpdateHit( GateCrystalHit* hit, const G4bool use_interactions_statistics, const std::map<G4int,InteractionsInfo>& gammas_track_id, const int eventID ); 

 private:
  std::unique_ptr<GateExtendedAnalysisMessenger> pMessenger;
  std::unique_ptr<GateTrajectoryNavigator> pTrajectoryNavigator;
  const G4String m_noFileName = "  ";// 2 spaces for output module with no fileName
};

#endif
