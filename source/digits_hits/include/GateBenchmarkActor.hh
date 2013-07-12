#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

#ifndef GATEBENCHMARKACTOR_HH
#define GATEBENCHMARKACTOR_HH

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

#include "GateVActor.hh"
#include "GateActorMessenger.hh"

class GateBenchmarkActor : public GateVActor
{
	public:

		virtual ~GateBenchmarkActor();

		FCT_FOR_AUTO_CREATOR_ACTOR(GateBenchmarkActor)

			virtual void Construct();

		virtual void BeginOfRunAction(const G4Run * r);
		virtual void BeginOfEventAction(const G4Event *) ;
		virtual void UserSteppingAction(const GateVVolume *, const G4Step*);

		virtual void PreUserTrackingAction(const GateVVolume *, const G4Track*) ;
		virtual void PostUserTrackingAction(const GateVVolume *, const G4Track*) ;
		virtual void EndOfEventAction(const G4Event*);

		virtual void SaveData();
		virtual void ResetData();

		virtual void clear(){ResetData();}
		virtual void Initialize(G4HCofThisEvent*){}
		virtual void EndOfEvent(G4HCofThisEvent*){}

	protected:
		GateBenchmarkActor(G4String name, G4int depth=0);

		TFile* pTfile;
		TH2D* histoEFreePath;
		TH2D* histoEStepLength;
		TH2D* histoEDeltaE;
		TH2D* histoEPrimaryDeviation;
		TH2D* histoESecondaryDeviation;
		TH1D* histoFlyDistance;
		TH1D* histoSumFreePath;

		G4ThreeVector positionInitial;
		G4double sumFreePath;
		size_t currentSecondary;

		GateActorMessenger* pMessenger;
};

MAKE_AUTO_CREATOR_ACTOR(BenchmarkActor,GateBenchmarkActor)

#endif
#endif
