/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#include "GateExtendedAnalysis.hh"

#include "globals.hh"
#include "G4Run.hh"
#include "G4Step.hh"
#include "G4Event.hh"
#include "G4VHitsCollection.hh"
#include "G4HCofThisEvent.hh"
#include "G4TrajectoryContainer.hh"
#include "G4VProcess.hh"
#include "G4Navigator.hh"
#include "G4TransportationManager.hh"

#include "GateSourceMgr.hh"
#include "GateCrystalHit.hh"
#include "GatePhantomHit.hh"
#include "GateTrajectoryNavigator.hh"
#include "GateOutputMgr.hh"
#include "GateActions.hh"

GateExtendedAnalysis::GateExtendedAnalysis(const G4String& name, GateOutputMgr* outputMgr,DigiMode digiMode) : GateVOutputModule(name,outputMgr,digiMode)
{
 m_isEnabled = false;
 pTrajectoryNavigator.reset( new GateTrajectoryNavigator() );
 pMessenger.reset( new GateExtendedAnalysisMessenger( this ) );
}
  
const G4String& GateExtendedAnalysis::GiveNameOfFile()
{
  return m_noFileName;
}

void GateExtendedAnalysis::RecordBeginOfAcquisition() {}

void GateExtendedAnalysis::RecordEndOfAcquisition() {}

void GateExtendedAnalysis::RecordBeginOfRun(const G4Run * ) {}

void GateExtendedAnalysis::RecordEndOfRun(const G4Run * ) {}

void GateExtendedAnalysis::RecordBeginOfEvent(const G4Event * ) {}

void GateExtendedAnalysis::RecordStepWithVolume(const GateVVolume *, const G4Step * ) {}

void GateExtendedAnalysis::RecordVoxels(GateVGeometryVoxelStore *) {}

std::map<G4int,GateExtendedAnalysis::InteractionsInfo> GateExtendedAnalysis::GetPositroniumGammasID( GateCrystalHitsCollection* chc)
{
 G4int nhits = chc->entries();
 std::map<G4int,GateExtendedAnalysis::InteractionsInfo> tracks_id;
 
 for ( G4int i = 0; i < nhits; ++i )
 {
  GateCrystalHit* hit = (*chc)[i];
  G4int id = hit->GetTrackID();
  G4int gamma_kind = hit->GetGammaKind();
  
  if ( gamma_kind > 0 )
  {
   std::map<G4int,InteractionsInfo>::iterator found = tracks_id.find( id );
   if ( found == tracks_id.end() ) { tracks_id.emplace( id, InteractionsInfo() ); }
  }
 }
 
 return tracks_id;
}

G4bool GateExtendedAnalysis::IsPositroniumGamma( const G4int track_id, const std::map<G4int,InteractionsInfo>& tracks_id )
{
 std::map<G4int,GateExtendedAnalysis::InteractionsInfo>::const_iterator found = tracks_id.find( track_id );
 return found != tracks_id.cend();
}

G4String GateExtendedAnalysis::GetProcessVolumeName( const G4ThreeVector& hit_pos )
{
 G4Navigator* gNavigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
 G4ThreeVector null(0.,0.,0.);
 return gNavigator->LocateGlobalPointAndSetup( hit_pos, &null, false )->GetName();
}

void GateExtendedAnalysis::FillPhantomInteractions( std::map<G4int,InteractionsInfo>& tracks_id )
{
 GatePhantomHitsCollection* PHC = GetOutputMgr()->GetPhantomHitCollection();
 
 if ( PHC == nullptr ) { return; }
 
 G4int n_hits = PHC->entries();
 
 for ( G4int i = 0; i < n_hits; ++i )
 {
  GatePhantomHit* hit = (*PHC)[i];
  G4int phantom_trackID = hit->GetTrackID();
  
  if ( !IsPositroniumGamma( phantom_trackID, tracks_id ) ) { continue; }
  
  G4String process_name = hit->GetProcess();
  G4ThreeVector hit_pos = hit->GetPos();
  
   if ( process_name.find("ompt") != G4String::npos )
   {
    ++tracks_id[phantom_trackID].n_phantom_compton;
    tracks_id[phantom_trackID].compton_volume_name = GetProcessVolumeName( hit_pos );
   }
   else if ( process_name.find("Rayl") != G4String::npos )
   {
    ++tracks_id[phantom_trackID].n_phantom_rayleigh;
    tracks_id[phantom_trackID].rayleigh_volume_name = GetProcessVolumeName( hit_pos );
   }  
 }
}

bool GateExtendedAnalysis::EventHasSources()
{
 return (GateSourceMgr::GetInstance())->GetSourcesForThisEvent().size() != 0;
}

void GateExtendedAnalysis::UpdateHit( GateCrystalHit* hit, const G4bool use_interactions_statistics, const std::map<G4int,InteractionsInfo>& gammas_track_id, const int eventID )
{
 if ( !hit->GoodForAnalysis() ) { return; }
 
 G4int trackID = hit->GetTrackID();
 G4int n_phantom_compton = 0;
 G4int n_crystal_compton = 0;
 G4int n_phantom_rayleigh = 0;
 G4int n_crystal_rayleigh = 0;
 G4String compton_volume_name("NULL");
 G4String rayleigh_volume_name("NULL");
 G4int primaryID  = pTrajectoryNavigator->FindPrimaryID( trackID );
 G4int runID   = GateRunManager::GetRunManager()->GetCurrentRun()->GetRunID();
 G4int sourceID = (((GateSourceMgr::GetInstance())->GetSourcesForThisEvent())[0])->GetSourceID();
 G4ThreeVector source_vertex = pTrajectoryNavigator->FindSourcePosition();
    
 //If this is a gamma emitted from positronium then save statistics
 if ( use_interactions_statistics )
 {
  n_phantom_compton = gammas_track_id.at(trackID).n_phantom_compton;
  n_crystal_compton = gammas_track_id.at(trackID).n_crystal_compton;
  n_phantom_rayleigh = gammas_track_id.at(trackID).n_phantom_rayleigh;
  n_crystal_rayleigh = gammas_track_id.at(trackID).n_crystal_rayleigh;
  compton_volume_name = gammas_track_id.at(trackID).compton_volume_name;
  rayleigh_volume_name = gammas_track_id.at(trackID).rayleigh_volume_name;
 }
    
 hit->SetSourceID( sourceID );
 hit->SetSourcePosition( source_vertex );
 hit->SetNPhantomCompton( n_phantom_compton);
 hit->SetNPhantomRayleigh( n_phantom_rayleigh);
 hit->SetComptonVolumeName ( compton_volume_name );
 hit->SetRayleighVolumeName( rayleigh_volume_name );
 hit->SetPrimaryID( primaryID );
 hit->SetEventID( eventID );
 hit->SetRunID( runID );
 hit->SetNCrystalCompton(n_crystal_compton);
 hit->SetNCrystalRayleigh( n_crystal_rayleigh ); 
}

void GateExtendedAnalysis::RecordEndOfEvent(const G4Event * event ) 
{
 G4TrajectoryContainer* trajectory_container = event->GetTrajectoryContainer();
 G4int eventID = event->GetEventID();

 if ( trajectory_container == nullptr )
 {
  if (nVerboseLevel > 0) { G4cout << "GateExtendedAnalysis::RecordEndOfEvent : WARNING : G4TrajectoryContainer not found\n"; }
  return;
 }
 else
 {
  pTrajectoryNavigator->SetTrajectoryContainer(trajectory_container);
  GateCrystalHitsCollection* CHC = GetOutputMgr()->GetCrystalHitCollection();
  
  if ( CHC == nullptr ) { return; }
  
  std::map<G4int,GateExtendedAnalysis::InteractionsInfo> gammas_track_id = GetPositroniumGammasID( CHC );
  FillPhantomInteractions( gammas_track_id );
  
  if ( !EventHasSources() ) { return; }
  
  G4int n_hits = CHC->entries();
  
  for ( G4int i = 0; i < n_hits; ++i )
  {
   GateCrystalHit* hit = (*CHC)[i];
   G4int trackID = hit->GetTrackID();
   G4String process_name = hit->GetProcess();
   G4bool use_interactions_statistics = false;
   
   if ( IsPositroniumGamma( trackID, gammas_track_id ) )
   {
    use_interactions_statistics = true;
    if ( process_name.find("ompt") != G4String::npos ) { ++gammas_track_id[trackID].n_crystal_compton; }
    else if ( process_name.find("Rayl") != G4String::npos ) { ++gammas_track_id[trackID].n_crystal_rayleigh; }
   }
   
   UpdateHit( hit, use_interactions_statistics, gammas_track_id, eventID );
  }
 }
}
