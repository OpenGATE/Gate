/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
#include "GateBenchmarkActor.hh"
#ifdef G4ANALYSIS_USE_ROOT

#include <G4VProcess.hh>

#include "GateBenchmarkActorMessenger.hh"

GateBenchmarkActor::GateBenchmarkActor(G4String name, G4int depth)
  : GateVActor(name,depth)
{
  pMessenger = new GateBenchmarkActorMessenger(this);
}

GateBenchmarkActor::~GateBenchmarkActor()
{
  delete pMessenger;
}

void GateBenchmarkActor::Construct()
{
  const int nbins = 1024;
  const G4double max_fly_distance = 15*mm;
  const G4double max_energy = 10*MeV;

  GateVActor::Construct();

  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(false);
  EnableEndOfEventAction(true); // for save every n
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);

  pTfile = new TFile(mSaveFilename,"RECREATE");

  histoEFreePath = new TH2D("EFreePath","Free path", nbins, 0, max_energy/MeV, nbins, 0, max_fly_distance/mm);
  histoEFreePath->SetXTitle("Energy [MeV]");
  histoEFreePath->SetYTitle("Distance [mm]");

  histoEStepLength = new TH2D("EStepLength","Step length", nbins, 0, max_energy/MeV, nbins, 0, max_fly_distance/mm);
  histoEStepLength->SetXTitle("Energy [MeV]");
  histoEStepLength->SetYTitle("Distance [mm]");

  histoEDeltaE = new TH2D("EDeltaE","Energy loss", nbins, 0, max_energy/MeV, nbins, 0, 2);
  histoEDeltaE->SetXTitle("Energy [MeV]");
  histoEDeltaE->SetYTitle("Energy loss [MeV]");

  histoEPrimaryDeviation = new TH2D("EPrimaryDeviation","Primary deviation angle", nbins, 0, max_energy/MeV, nbins, 0, 2);
  histoEPrimaryDeviation->SetXTitle("Energy [MeV]");
  histoEPrimaryDeviation->SetYTitle("Deviation angle [deg]");

  histoESecondaryDeviation = new TH2D("ESecondaryDeviation","Secondary deviation angle", nbins, 0, max_energy/MeV, nbins, 40, 90);
  histoESecondaryDeviation->SetXTitle("Energy [MeV]");
  histoESecondaryDeviation->SetYTitle("Deviation angle [deg]");

  histoFlyDistance = new TH1D("FlyDistance","Fly distance per track", nbins, 0, max_fly_distance/mm);
  histoFlyDistance->SetXTitle("Distance [mm]");

  histoSumFreePath = new TH1D("SumFreePath","Sum of free path per track", nbins, 0, max_fly_distance/mm);
  histoSumFreePath->SetXTitle("Distance [mm]");

  ResetData();
}

void GateBenchmarkActor::SaveData()
{
  GateVActor::SaveData();
  pTfile->Write();
}

void GateBenchmarkActor::ResetData()
{
  histoEFreePath->Reset();
  histoEFreePath->Reset();
  histoEDeltaE->Reset();
  histoEPrimaryDeviation->Reset();
  histoESecondaryDeviation->Reset();
  histoFlyDistance->Reset();
  histoSumFreePath->Reset();
}

void GateBenchmarkActor::BeginOfRunAction(const G4Run*)
{
  GateDebugMessage("Actor", 3, "GateBenchmarkActor -- Begin of Run" << G4endl);
}

void GateBenchmarkActor::BeginOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateBenchmarkActor -- Begin of Event" << G4endl);
}

void GateBenchmarkActor::EndOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateBenchmarkActor -- End of Event" << G4endl);
}

void GateBenchmarkActor::PreUserTrackingAction(const GateVVolume*, const G4Track* track)
{
  const G4String name = track->GetDefinition()->GetParticleName();
  const G4ThreeVector position = track->GetPosition();
  //const G4double energy = track->GetKineticEnergy();

  //G4cout << "begin track for " << name << " position = " << position/mm << " energy = " << energy/MeV << G4endl;

  positionInitial = position;
  sumFreePath = 0;
  currentSecondary = 0;
}

G4double deviationAngle(const G4ThreeVector& dir_orig, const G4ThreeVector& dir_final)
{
  G4double dot_product = dir_orig.dot(dir_final);
  if (dot_product > 1) dot_product = 1;
  if (dot_product < -1) dot_product = -1;
  return acos(dot_product)*rad;
}

void GateBenchmarkActor::UserSteppingAction(const GateVVolume*, const G4Step* step)
{
  const G4double weight = step->GetTrack()->GetWeight();

  const G4ThreeVector position_pre = step->GetPreStepPoint()->GetPosition();
  const G4ThreeVector position_post = step->GetPostStepPoint()->GetPosition();
  const G4double step_length = step->GetStepLength();
  const G4double free_path = (position_post-position_pre).mag();

  const G4double energy_pre = step->GetPreStepPoint()->GetKineticEnergy();
  const G4double energy_post = step->GetPostStepPoint()->GetKineticEnergy();
  const G4double energy = energy_pre;
  const G4double energy_delta = energy_pre-energy_post;

  const G4ThreeVector direction_pre = step->GetPostStepPoint()->GetMomentumDirection();
  const G4ThreeVector direction_post = step->GetPostStepPoint()->GetMomentumDirection();
  const G4double deviation = deviationAngle(direction_post,direction_pre);

  const G4VProcess* process = step->GetPostStepPoint()->GetProcessDefinedStep();
  G4String process_name = "unknown";
  if (process) process_name = process->GetProcessName();
  if (process_name != "ElectronIonisation") return;

  sumFreePath += free_path;

  //G4cout << "step position_pre = " << position_pre/mm << " position_post = " << position_post/mm << " free_path = " << free_path/mm << G4endl;
  //G4cout << "     energy_pre = " << energy_pre/MeV << " energy_post = " << energy_post/MeV << " delta = " << energy_delta/MeV << G4endl;
  //G4cout << "     direction_pre = " << direction_pre << " direction_post = " << direction_post << " deviation = " << deviation/deg << G4endl;
  //G4cout << "     process = " << process_name << G4endl;

  histoEFreePath->Fill(energy/MeV,free_path/mm,weight);
  histoEStepLength->Fill(energy/MeV,step_length/mm,weight);
  histoEDeltaE->Fill(energy/MeV,energy_delta/MeV,weight);
  if (deviation) histoEPrimaryDeviation->Fill(energy/MeV,deviation/MeV,weight);

  //G4cout << "     nsec = " << step->GetSecondary()->size() << G4endl;
  while (currentSecondary < step->GetSecondary()->size())
	{
      const G4Track* track_secondary = (*step->GetSecondary())[currentSecondary];
      const G4ThreeVector direction_secondary = track_secondary->GetMomentumDirection();
      const G4double deviation_secondary = deviationAngle(direction_secondary,direction_pre);

      //G4cout << "         currentSecondary = " << currentSecondary << " direction_pre = " << direction_pre << " direction_sec = " << direction_secondary << " deviation = " << deviation_secondary/deg << G4endl;

      if (deviation_secondary) histoESecondaryDeviation->Fill(energy/MeV,deviation_secondary/deg,weight);

      currentSecondary++;
	}
}

void GateBenchmarkActor::PostUserTrackingAction(const GateVVolume*, const G4Track* track)
{
  const G4ThreeVector position = track->GetPosition();
  const G4double fly_distance = (position-positionInitial).mag();
  const G4double weight = track->GetWeight();

  //G4cout << "end track position = " << position/mm << " flyDistance = " << fly_distance/mm << " sumFreePath = " << sumFreePath/mm << G4endl;

  histoFlyDistance->Fill(fly_distance/mm,weight);
  histoSumFreePath->Fill(sumFreePath/mm,weight);
}

#endif
