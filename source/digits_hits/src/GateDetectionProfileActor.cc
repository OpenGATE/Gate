/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GATEDETECTIONPROFILEACTOR_CC
#define GATEDETECTIONPROFILEACTOR_CC

#include "GateDetectionProfileActor.hh"
#include "GateMiscFunctions.hh"

GateDetectionProfileActor::GateDetectionProfileActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateDetectionProfileActor() -- begin"<<G4endl);
  pMessenger = new GateDetectionProfileActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateDetectionProfileActor() -- end"<<G4endl);
}


GateDetectionProfileActor::~GateDetectionProfileActor()
{
  delete pMessenger;
}

void GateDetectionProfileActor::Construct()
{
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);
  EnableEndOfEventAction(true); // for save every n

  //mHistName = "Precise/output/EnergySpectrum.root";
  //pTfile = new TFile(mSaveFilename,"RECREATE");

  //pEnergySpectrum = new TH1D("energySpectrum","Energy Spectrum",GetENBins(),GetEmin() ,GetEmax() );
  //pEnergySpectrum->SetXTitle("Energy (MeV)");

  //pEdep  = new TH1D("edepHisto","Energy deposited",GetEdepNBins(),GetEdepmin() ,GetEdepmax() );
  //pEdep->SetXTitle("E_{dep} (MeV)");

  ResetData();
}

void GateDetectionProfileActor::SaveData()
{
  GateMessage("Actor", 0, "GateDetectionProfileActor -- Saving data" << G4endl);
  //pTfile->Write();
  //pTfile->Close();
}

void GateDetectionProfileActor::ResetData() 
{
}

void GateDetectionProfileActor::BeginOfRunAction(const G4Run *)
{
  GateDebugMessage("Actor", 3, "GateDetectionProfileActor -- Begin of Run" << G4endl);
}

void GateDetectionProfileActor::BeginOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateDetectionProfileActor -- Begin of Event" << G4endl);
}

void GateDetectionProfileActor::EndOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateDetectionProfileActor -- End of Event" << G4endl);
}

void GateDetectionProfileActor::PreUserTrackingAction(const GateVVolume *, const G4Track* t) 
{
  GateDebugMessage("Actor", 3, "GateDetectionProfileActor -- Begin of Track" << G4endl);
}

void GateDetectionProfileActor::PostUserTrackingAction(const GateVVolume *, const G4Track* t) 
{
  GateDebugMessage("Actor", 3, "GateDetectionProfileActor -- End of Track" << G4endl);
}

void GateDetectionProfileActor::UserSteppingAction(const GateVVolume *, const G4Step* step)
{
}

GateDetectionProfilePrimaryTimerActor::GateDetectionProfilePrimaryTimerActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateDetectionProfilePrimaryTimerActor() -- begin"<<G4endl);
  pMessenger = new GateDetectionProfilePrimaryTimerActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateDetectionProfilePrimaryTimerActor() -- end"<<G4endl);
}


GateDetectionProfilePrimaryTimerActor::~GateDetectionProfilePrimaryTimerActor()
{
  delete pMessenger;
}

void GateDetectionProfilePrimaryTimerActor::Construct()
{
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);
  EnableEndOfEventAction(true); // for save every n

  //mHistName = "Precise/output/EnergySpectrum.root";
  //pTfile = new TFile(mSaveFilename,"RECREATE");

  //pEnergySpectrum = new TH1D("energySpectrum","Energy Spectrum",GetENBins(),GetEmin() ,GetEmax() );
  //pEnergySpectrum->SetXTitle("Energy (MeV)");

  //pEdep  = new TH1D("edepHisto","Energy deposited",GetEdepNBins(),GetEdepmin() ,GetEdepmax() );
  //pEdep->SetXTitle("E_{dep} (MeV)");

  ResetData();
}

void GateDetectionProfilePrimaryTimerActor::SaveData()
{
  GateMessage("Actor", 0, "GateDetectionProfilePrimaryTimerActor -- Saving data" << G4endl);
  //pTfile->Write();
  //pTfile->Close();
}

void GateDetectionProfilePrimaryTimerActor::ResetData() 
{
}

void GateDetectionProfilePrimaryTimerActor::BeginOfRunAction(const G4Run *)
{
  GateDebugMessage("Actor", 3, "GateDetectionProfilePrimaryTimerActor -- Begin of Run" << G4endl);
}

void GateDetectionProfilePrimaryTimerActor::BeginOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateDetectionProfilePrimaryTimerActor -- Begin of Event" << G4endl);
}

void GateDetectionProfilePrimaryTimerActor::EndOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateDetectionProfilePrimaryTimerActor -- End of Event" << G4endl);
}

void GateDetectionProfilePrimaryTimerActor::PreUserTrackingAction(const GateVVolume *, const G4Track* t) 
{
  GateDebugMessage("Actor", 3, "GateDetectionProfilePrimaryTimerActor -- Begin of Track" << G4endl);
}

void GateDetectionProfilePrimaryTimerActor::PostUserTrackingAction(const GateVVolume *, const G4Track* t) 
{
  GateDebugMessage("Actor", 3, "GateDetectionProfilePrimaryTimerActor -- End of Track" << G4endl);
}

void GateDetectionProfilePrimaryTimerActor::UserSteppingAction(const GateVVolume *, const G4Step* step)
{
}

#endif 
