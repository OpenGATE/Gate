/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
#include "GateAugerDetectorActor.hh"
#ifdef G4ANALYSIS_USE_ROOT

#include <Randomize.hh>
#include "GateAugerDetectorActorMessenger.hh"

//-----------------------------------------------------------------------------
GateAugerDetectorActor::GateAugerDetectorActor(G4String name, G4int depth)
  : GateVActor(name,depth)
{
  max_time_of_flight = 10*ns;
  min_energy_deposition = 2*MeV;
  projection_direction = G4ThreeVector(1,0,0);
  profile_min = -160*mm;
  profile_max = 160*mm;
  profile_nbpts = 361;
  profile_noise_fwhm = 5*mm;

  pMessenger = new GateAugerDetectorActorMessenger(this);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateAugerDetectorActor::~GateAugerDetectorActor()
{
  delete pMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::setMaxTOF(G4double tof)
{
  max_time_of_flight = tof;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::setMinEdep(G4double edep)
{
  min_energy_deposition = edep;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::setProjectionDirection(const G4ThreeVector& dir)
{
  projection_direction = dir;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::setMinimumProfileAxis(G4double min)
{
  profile_min = min;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::setMaximumProfileAxis(G4double max)
{
  profile_max = max;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::setProfileSize(int nbpts)
{
  profile_nbpts = nbpts;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::setProfileNoiseFWHM(G4double noise_fwhm)
{
  profile_noise_fwhm = noise_fwhm;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::Construct()
{
  GateVActor::Construct();

  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);
  EnableEndOfEventAction(true);

  pTfile = new TFile(mSaveFilename,"RECREATE");

  pProfileHisto = new TH1D("reconstructedProfileHisto","reconstructed profile",profile_nbpts,profile_min,profile_max);
  pProfileHisto->SetXTitle("position (mm)");

  pEnergyDepositionHisto = new TH1D("edepHisto","energy deposited",500,0,5);
  pEnergyDepositionHisto->SetXTitle("deposited energy (MeV)");

  pTimeOfFlightHisto = new TH1D("tofHisto","time of flight",500,0,max_time_of_flight);
  pTimeOfFlightHisto->SetXTitle("time of flight (ns)");

  ResetData();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::SaveData()
{
  GateVActor::SaveData(); // filename from current run/event not taken into account
  pTfile->Write();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::ResetData()
{
  pProfileHisto->Reset();
  pEnergyDepositionHisto->Reset();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::BeginOfRunAction(const G4Run*)
{
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::BeginOfEventAction(const G4Event*)
{
  depositions.clear();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::EndOfEventAction(const G4Event*)
{
  const G4double total_deposited_energy = GetTotalDepositedEnergy();

  if (total_deposited_energy <= 0) return;
  pEnergyDepositionHisto->Fill(total_deposited_energy/MeV);
  pTimeOfFlightHisto->Fill(GetWeighedBarycenterTime()/ns);

  if (total_deposited_energy < min_energy_deposition) return;
  const G4ThreeVector hit_position = GetWeighedBarycenterPosition();
  const G4double noise_projection = G4RandGauss::shoot(0,profile_noise_fwhm/2.3548);
  //G4cout << "HITTTTTED!!!!!" << G4endl;
  //G4cout << "ndep = " << depositions.size() << " total_edep = " << total_deposited_energy << G4endl;
  //G4cout << "position = " << hit_position << G4endl;
  pProfileHisto->Fill((projection_direction.dot(hit_position)+noise_projection)/mm);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::PreUserTrackingAction(const GateVVolume*, const G4Track*)
{
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::PostUserTrackingAction(const GateVVolume*, const G4Track*)
{
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::UserSteppingAction(const GateVVolume*, const G4Step* step)
{
  const G4double time = step->GetPostStepPoint()->GetGlobalTime();
  if (time>max_time_of_flight) return;

  AugerDeposition deposition;
  deposition.position = (step->GetPostStepPoint()->GetPosition()+step->GetPreStepPoint()->GetPosition())/2.;
  deposition.energy = step->GetTotalEnergyDeposit();
  deposition.time = time;
  if (deposition.energy <= 0) return;

  depositions.push_back(deposition);

  //G4cout << "edep = " << deposition.energy/MeV << " " << (step->GetPreStepPoint()->GetKineticEnergy()-step->GetPostStepPoint()->GetKineticEnergy())/MeV << G4endl;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4double GateAugerDetectorActor::GetTotalDepositedEnergy() const
{
  G4double total_deposited_energy = 0;
  for (std::list<AugerDeposition>::const_iterator iter=depositions.begin(); iter!=depositions.end(); iter++)
	{
      total_deposited_energy += iter->energy;
	}
  return total_deposited_energy;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4ThreeVector GateAugerDetectorActor::GetWeighedBarycenterPosition() const
{
  G4double total_weight = 0;
  G4ThreeVector accum(0,0,0);
  for (std::list<AugerDeposition>::const_iterator iter=depositions.begin(); iter!=depositions.end(); iter++)
	{
      total_weight += iter->energy;
      accum += iter->position*iter->energy;
	}
  return accum/total_weight;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4double GateAugerDetectorActor::GetWeighedBarycenterTime() const
{
  G4double total_weight = 0;
  G4double accum = 0;
  for (std::list<AugerDeposition>::const_iterator iter=depositions.begin(); iter!=depositions.end(); iter++)
	{
      total_weight += iter->energy;
      accum += iter->time*iter->energy;
	}
  return accum/total_weight;
}
//-----------------------------------------------------------------------------

#endif
