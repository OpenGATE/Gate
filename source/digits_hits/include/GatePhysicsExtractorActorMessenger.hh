#ifndef GATEPHYSICSEXTRACTORACTORMESSENGER_HH
#define GATEPHYSICSEXTRACTORACTORMESSENGER_HH

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateActorMessenger.hh"

#include <G4UIcmdWithAString.hh>
#include <G4UIcmdWithADouble.hh>
#include <G4UIcmdWithADoubleAndUnit.hh>

class GatePhysicsExtractorActor;

class GatePhysicsExtractorActorMessenger : public GateActorMessenger
{
	public:
		GatePhysicsExtractorActorMessenger(GatePhysicsExtractorActor*);
		~GatePhysicsExtractorActorMessenger();

		void SetNewValue(G4UIcommand*, G4String);
	protected:
		void BuildCommands(G4String base);

		G4UIcmdWithAString* pProcessNameCmd;
		G4UIcmdWithAString* pParticleNameCmd;
		G4UIcmdWithAString* pMaterialNameCmd;
		G4UIcmdWithADouble* pNbPtsCmd;
		G4UIcmdWithADoubleAndUnit* pEminCmd;
		G4UIcmdWithADoubleAndUnit* pEmaxCmd;
		G4UIcmdWithADoubleAndUnit* pEnergyCutCmd;

		GatePhysicsExtractorActor* pActor;
};

#endif
#endif
