#ifndef GATEGPSPECTRUMACTOR_HH
#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

#include "GateVActor.hh"
#include "GateActorMessenger.hh"

class GateGpSpectrumActor : public GateVActor
{
	public:
		virtual ~GateGpSpectrumActor();

		FCT_FOR_AUTO_CREATOR_ACTOR(GateGpSpectrumActor)

		virtual void Construct();

		virtual void BeginOfRunAction(const G4Run*);
		virtual void BeginOfEventAction(const G4Event*) ;
		virtual void UserSteppingAction(const GateVVolume*, const G4Step*);

		virtual void PreUserTrackingAction(const GateVVolume*, const G4Track*);
		virtual void PostUserTrackingAction(const GateVVolume*, const G4Track*);
		virtual void EndOfEventAction(const G4Event*);

		virtual void SaveData();
		virtual void ResetData();
	protected:
		GateGpSpectrumActor(G4String name, G4int depth=0);

		size_t last_secondaries_size;
		bool first_step;
		GateActorMessenger* pMessenger;
		TFile* pTfile;
		TH2D* pHEpEgp;
		TH2D* pHEpEgpNormalized;
		TH1D* pHEpInelastic;
		TH1D* pHEpInelasticProducedGamma;
};

MAKE_AUTO_CREATOR_ACTOR(GpSpectrumActor,GateGpSpectrumActor)

#endif
#endif
