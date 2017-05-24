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
#include "GateConstants.hh"
//-----------------------------------------------------------------------------
GateAugerDetectorActor::GateAugerDetectorActor(G4String name, G4int depth) :
    GateVActor(name, depth)
  {
  min_time_of_flight = 0 * ns;
  max_time_of_flight = 10 * ns;
  min_energy_deposition = 1 * MeV;
  max_energy_deposition = 8 * MeV;
  projection_direction = G4ThreeVector(1, 0, 0);
  profile_min = -160 * mm;
  profile_max = 160 * mm;
  profile_nbpts = 361;
  profile_noise_fwhm = 5 * mm;

  _isARF = false;
  _nbOfHeads = 1;
  _nbOfSourcePhotons = 0;
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
void GateAugerDetectorActor::setMinTOF(G4double tof)
  {
  min_time_of_flight = tof;
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
void GateAugerDetectorActor::setMaxEdep(G4double edep)
  {
  max_energy_deposition = edep;
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

void GateAugerDetectorActor::setGenerateArf(G4bool b)
  {
  _isARF = b;
  }

void GateAugerDetectorActor::setArfFilename(G4String s)
  {
  _arfDataFilename = s;
  }
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::setProfileNoiseFWHM(G4double noise_fwhm)
  {
  profile_noise_fwhm = noise_fwhm;
  }
//-----------------------------------------------------------------------------

void GateAugerDetectorActor::defineArfTRootFile()
  {
  _arfDataFile = new TFile(_arfDataFilename.c_str(), "RECREATE", "ROOT file for ARF purpose");
  _arfDataTree = new TTree("theTree", "ARF Data Tree");
  _arfDataTree->Branch("Edep", &_depositedEnergy, "Edep/D");
  _arfDataTree->Branch("outY", &_projectionPositionY, "outY/D");
  _arfDataTree->Branch("outX", &_projectionPositionX, "outX/D");
  _nbOfPhotonsTree = new TTree("theNumberOfPhoton", " statistics on Simulated Photons");
  _nbOfPhotonsTree->Branch("NbOfSourcePhot", &_nbOfSourcePhotons, "NbOfSourcePhotons/l");
  _nbOfPhotonsTree->Branch("NbOfHeads", &_nbOfHeads, "NbOfHeads/I");
  }

void GateAugerDetectorActor::closeArfDataRootFile()
  {
  _arfDataFile = _arfDataTree->GetCurrentFile();
  if (_arfDataFile->IsOpen())
    {
    _nbOfPhotonsTree->Fill();
    _arfDataFile->Write();
    _arfDataFile->Close();
    }
  }

void GateAugerDetectorActor::storeArfRootData(const G4ThreeVector direction, const G4double energy)
  {
  _nbOfStoredPhotons++;
  if (_nbOfStoredPhotons % 1000 == 0)
    {
    G4cout << " number of stored photons    " << _nbOfStoredPhotons << Gateendl;
    G4cout << " number of NbOfSourcePhotons " << _nbOfSourcePhotons << Gateendl;
    }

  _projectionPositionX = direction.x();
  _projectionPositionY = direction.y();

  _depositedEnergy = energy;
  _arfDataTree->Fill();
  }

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

  pTfile = new TFile(mSaveFilename, "RECREATE");

  pProfileHisto = new TH1D("reconstructedProfileHisto",
                           "reconstructed profile",
                           profile_nbpts,
                           profile_min,
                           profile_max);
  pProfileHisto->SetXTitle("position (mm)");

  pEnergyDepositionHisto = new TH1D("edepHisto", "energy deposited", 500, 0, 5);
  pEnergyDepositionHisto->SetXTitle("deposited energy (MeV)");

  pTimeOfFlightHisto = new TH1D("tofHisto",
                                "time of flight",
                                500,
                                min_time_of_flight,
                                max_time_of_flight);
  pTimeOfFlightHisto->SetXTitle("time of flight (ns)");

  if (_isARF)
    {
    defineArfTRootFile();
    }
  ResetData();

  }
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::SaveData()
  {
  GateVActor::SaveData(); // filename from current run/event not taken into account
  pTfile->Write();
  if (_isARF)
    {
    closeArfDataRootFile();
    }

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
void GateAugerDetectorActor::EndOfEventAction(const G4Event* e)
  {
  const G4double total_deposited_energy = GetTotalDepositedEnergy();
  _nbOfSourcePhotons += 1;
  if (total_deposited_energy <= 0)
    {
    return;
    }
  pEnergyDepositionHisto->Fill(total_deposited_energy / MeV);
  pTimeOfFlightHisto->Fill(GetWeighedBarycenterTime() / ns);

  if (total_deposited_energy < min_energy_deposition)
    {
    return;
    }
  if (total_deposited_energy > max_energy_deposition)
    {
    return;
    }
  if (_isARF)
    {
    GateAugerDetectorActor::storeArfRootData(GetWeighedBarycenterPosition(),
                                             e->GetPrimaryVertex()->GetPrimary()->GetTotalEnergy());
    }
  const G4ThreeVector hit_position = GetWeighedBarycenterPosition();
  const G4double noise_projection = G4RandGauss::shoot(0,profile_noise_fwhm/GateConstants::fwhm_to_sigma);
  double pos = (projection_direction.dot(hit_position) + noise_projection) / mm;
  pProfileHisto->Fill(pos);
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
  G4double timeOfFlight;
  if (_isARF)
    {
    timeOfFlight = step->GetPostStepPoint()->GetLocalTime();
    }
  else
    {
    timeOfFlight = step->GetPostStepPoint()->GetGlobalTime();
    }
  if (timeOfFlight < min_time_of_flight)
    {
    return;
    }
  if (timeOfFlight > max_time_of_flight)
    {
    return;
    }

  //2016-02-16: For LOCAL coords a transform MUST be made, see PhaseSpaceActor.cc:309-312
  //also see http://geant4.cern.ch/support/faq.shtml#a-geom-4
  G4ThreeVector worldPosition = (step->GetPostStepPoint()->GetPosition()
                                 + step->GetPreStepPoint()->GetPosition())
                                * 0.5;
  const G4AffineTransform transformation = step->GetPreStepPoint()->GetTouchable()->GetHistory()->GetTopTransform();
  G4ThreeVector localPosition = transformation.TransformPoint(worldPosition);

  //DD(worldPosition.x());
  //DD(localPosition.x());

  AugerDeposition deposition;
  deposition.position = localPosition;
  deposition.energy = step->GetTotalEnergyDeposit();
  deposition.time = timeOfFlight;
  if (deposition.energy <= 0)
    {
    return;
    }
  depositions.push_back(deposition);

  //G4cout << "edep = " << deposition.energy/MeV << " " << (step->GetPreStepPoint()->GetKineticEnergy()-step->GetPostStepPoint()->GetKineticEnergy())/MeV << Gateendl;
  }
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4double GateAugerDetectorActor::GetTotalDepositedEnergy() const
  {
  G4double total_deposited_energy = 0;
  for (std::list<AugerDeposition>::const_iterator iter = depositions.begin();
      iter != depositions.end(); iter++)
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
  G4ThreeVector accum(0, 0, 0);
  for (std::list<AugerDeposition>::const_iterator iter = depositions.begin();
      iter != depositions.end(); iter++)
    {
    total_weight += iter->energy;
    accum += iter->position * iter->energy;
    }
  return accum / total_weight;
  }










//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4double GateAugerDetectorActor::GetWeighedBarycenterTime() const
  {
  G4double total_weight = 0;
  G4double accum = 0;
  for (std::list<AugerDeposition>::const_iterator iter = depositions.begin();
      iter != depositions.end(); iter++)
    {
    total_weight += iter->energy;
    accum += iter->time * iter->energy;
    }
  return accum / total_weight;
  }
//-----------------------------------------------------------------------------

#endif
