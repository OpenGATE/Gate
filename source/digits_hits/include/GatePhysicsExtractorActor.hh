#ifndef GATEPHYSICSEXTRACTORACTOR_HH
#define GATEPHYSICSEXTRACTORACTOR_HH

#include "GateConfiguration.h"

#include "GateVActor.hh"
#include "GatePhysicsExtractorActorMessenger.hh"

#include <G4EmCalculator.hh>
#include <G4ParticleDefinition.hh>
#include <G4Material.hh>

class GatePhysicsExtractorActor : public GateVActor
{
	public:
		virtual ~GatePhysicsExtractorActor();

		FCT_FOR_AUTO_CREATOR_ACTOR(GatePhysicsExtractorActor)

		virtual void Construct();

		virtual void BeginOfRunAction(const G4Run*);
		virtual void BeginOfEventAction(const G4Event*) ;
		virtual void UserSteppingAction(const GateVVolume*, const G4Step*);

		virtual void PreUserTrackingAction(const GateVVolume*, const G4Track*);
		virtual void PostUserTrackingAction(const GateVVolume*, const G4Track*);
		virtual void EndOfEventAction(const G4Event*);

		virtual void SaveData();
		virtual void ResetData();

		inline void SetProcessName(const G4String& name) { pProcessName = name; };
		void SetParticleName(const G4String& particleName);
		void SetMaterialName(const G4String& materialName);
		inline void SetEnergyCutThreshold(double thresh) { pCutThreshold = thresh; };
		inline void SetMinEnergy(double energy) { pMinEnergy = energy; };
		inline void SetMaxEnergy(double energy) { pMaxEnergy = energy; };
		inline void SetNumberOfPointsPerOrderOfMagnitude(double nbPoints) { pNbPoints = nbPoints; };
	protected:
		GatePhysicsExtractorActor(G4String name, G4int depth=0);

		G4EmCalculator* pCalculator;
		const G4ParticleDefinition* pParticleDefinition;
		const G4Material* pMaterial;

		G4String pProcessName;
		double pMinEnergy;
		double pMaxEnergy;
		double pNbPoints;
		double pCutThreshold;

		GatePhysicsExtractorActorMessenger* pMessenger;
};

MAKE_AUTO_CREATOR_ACTOR(PhysicsExtractorActor,GatePhysicsExtractorActor)

#endif
