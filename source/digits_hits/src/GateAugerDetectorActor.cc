/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/
#ifdef G4ANALYSIS_USE_ROOT

/*
  \brief Class GateAugerDetectorActor : 
  \brief 
 */

#ifndef GATEAUGERDETECTORACTOR_CC
#define GATEAUGERDETECTORACTOR_CC

#include "GateAugerDetectorActor.hh"

#include "GateMiscFunctions.hh"
#include <iostream>
using std::endl;
using std::cout;

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateAugerDetectorActor::GateAugerDetectorActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  max_time_of_flight = 10*ns;
  min_energy_deposition = 2*MeV;

  pMessenger = new GateAugerDetectorActorMessenger(this);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor 
GateAugerDetectorActor::~GateAugerDetectorActor() 
{
}
//-----------------------------------------------------------------------------

void GateAugerDetectorActor::setMaxTOF(G4double tof)
{
	max_time_of_flight = tof;
}

void GateAugerDetectorActor::setMinEdep(G4double edep)
{
	min_energy_deposition = edep;
}

//-----------------------------------------------------------------------------
/// Construct
void GateAugerDetectorActor::Construct()
{
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);
  EnableEndOfEventAction(true); // for save every n

  pTfile = new TFile(mSaveFilename,"RECREATE");

  pProfileHisto = new TH1D("reconstructedProfileHisto","reconstructed profile",500,-200,200);
  pProfileHisto->SetXTitle("position (mm)");

  pEnergyDepositionHisto  = new TH1D("edepHisto","energy deposited",500,0,5);
  pEnergyDepositionHisto->SetXTitle("deposited energy (MeV)");
  
  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateAugerDetectorActor::SaveData()
{
  pTfile->Write();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateAugerDetectorActor::ResetData() 
{
	//->Reset();

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::BeginOfRunAction(const G4Run *)
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

	if (total_deposited_energy < min_energy_deposition) return;
	const G4ThreeVector hit_position = GetWeighedBarycenter();
	//G4cout << "HITTTTTED!!!!!" << G4endl;
	//G4cout << "ndep = " << depositions.size() << " total_edep = " << total_deposited_energy << G4endl;
	//G4cout << "position = " << hit_position << G4endl;
	pProfileHisto->Fill(hit_position[0]/mm);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::PreUserTrackingAction(const GateVVolume *, const G4Track* track) 
{
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActor::PostUserTrackingAction(const GateVVolume *, const G4Track* track) 
{
}
//-----------------------------------------------------------------------------

//G4bool GateAugerDetectorActor::ProcessHits(G4Step * step , G4TouchableHistory* /*th*/)
void GateAugerDetectorActor::UserSteppingAction(const GateVVolume *, const G4Step* step)
{
	const G4double time = step->GetPostStepPoint()->GetGlobalTime();
	if (time>max_time_of_flight) return;

	AugerDeposition deposition;
	deposition.position = (step->GetPostStepPoint()->GetPosition()+step->GetPreStepPoint()->GetPosition())/2.;
	deposition.deposited_energy = step->GetTotalEnergyDeposit();
	deposition.deposition_time = time;
	if (deposition.deposited_energy <= 0) return;

	depositions.push_back(deposition);
	
	//G4cout << "edep = " << deposition.deposited_energy/MeV << " " << (step->GetPreStepPoint()->GetKineticEnergy()-step->GetPostStepPoint()->GetKineticEnergy())/MeV << G4endl;
}
//-----------------------------------------------------------------------------


G4double GateAugerDetectorActor::GetTotalDepositedEnergy() const 
{
	G4double total_deposited_energy = 0;
	for (std::list<AugerDeposition>::const_iterator iter=depositions.begin(); iter!=depositions.end(); iter++)
	{
		total_deposited_energy += iter->deposited_energy;
	}
	return total_deposited_energy;
}

G4ThreeVector GateAugerDetectorActor::GetWeighedBarycenter() const 
{
	G4double total_weight = 0;
	G4ThreeVector accum(0,0,0);
	for (std::list<AugerDeposition>::const_iterator iter=depositions.begin(); iter!=depositions.end(); iter++)
	{
		total_weight += iter->deposited_energy;
		accum += iter->position*iter->deposited_energy;
	}
	return accum/total_weight;
}

#endif /* end #define GATEAUGERDETECTORACTOR_CC */
#endif
