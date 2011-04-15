/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GATEDETECTIONPROFILEACTOR_CC
#define GATEDETECTIONPROFILEACTOR_CC

#ifdef G4ANALYSIS_USE_ROOT

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
  messenger = new GateDetectionProfilePrimaryTimerActorMessenger(this);
  rootFile  = NULL;
  triggered = false;
}

GateDetectionProfilePrimaryTimerActor::~GateDetectionProfilePrimaryTimerActor()
{
  delete messenger;
}

void GateDetectionProfilePrimaryTimerActor::Construct()
{
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);
  EnableEndOfEventAction(false);

  rootFile = new TFile(mSaveFilename,"RECREATE");

  histoTime = new TH1D("triggerTime","Trigger Time",100,0.,1.);
  histoTime->SetXTitle("time [ns]");

  histoPosition = new TH2D("triggerPosition","Trigger Position",101,-5.,5.,101,-5.,5.);
  histoPosition->SetXTitle("x [mm]");
  histoPosition->SetYTitle("y [mm]");

  histoDirz = new TH1D("triggerDirection","Trigger Direction",100,.5,1.5);

  ResetData();
}

void GateDetectionProfilePrimaryTimerActor::SaveData()
{
  rootFile->Write();
  rootFile->Close();
}

void GateDetectionProfilePrimaryTimerActor::ResetData() 
{
  histoTime->Reset();
  histoPosition->Reset();
}

void GateDetectionProfilePrimaryTimerActor::BeginOfRunAction(const G4Run*)
{
}

void GateDetectionProfilePrimaryTimerActor::BeginOfEventAction(const G4Event*)
{
  triggered = false;
}

void GateDetectionProfilePrimaryTimerActor::EndOfEventAction(const G4Event*)
{
}

void GateDetectionProfilePrimaryTimerActor::PreUserTrackingAction(const GateVVolume*, const G4Track*) 
{
}

void GateDetectionProfilePrimaryTimerActor::PostUserTrackingAction(const GateVVolume*, const G4Track*) 
{
}

void GateDetectionProfilePrimaryTimerActor::UserSteppingAction(const GateVVolume*, const G4Step *step)
{
  if (triggered) return;
  if (!step->IsLastStepInVolume()) return;

  const G4StepPoint *point = step->GetPostStepPoint();

  triggered = true;
  data.name = step->GetTrack()->GetDefinition()->GetParticleName();
  data.position = point->GetPosition();
  data.direction = point->GetMomentumDirection();
  data.time = point->GetGlobalTime();
  G4double weight = point->GetWeight();

  histoTime->Fill(data.time,weight);
  histoPosition->Fill(data.position[0],data.position[1],weight);
  histoDirz->Fill(data.direction[2],weight);

  GateMessage("Actor",4,"triggered by " << data.name << " at " << data.time/ns << "ns " << data.position/mm << "mm" << G4endl);
}

#endif 
#endif 
