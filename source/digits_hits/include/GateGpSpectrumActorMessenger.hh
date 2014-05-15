#ifndef GATEGPSPECTRUMACTORMESSENGER_HH
#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

#include "GateActorMessenger.hh"

#include <G4UIcmdWithAnInteger.hh>
#include <G4UIcmdWithADoubleAndUnit.hh>

class GateGpSpectrumActor;

class GateGpSpectrumActorMessenger : public GateActorMessenger
{
	public:

		GateGpSpectrumActorMessenger(GateGpSpectrumActor*);
		~GateGpSpectrumActorMessenger();

		void SetNewValue(G4UIcommand*, G4String);
	protected:
		void BuildCommands(G4String base);
		GateGpSpectrumActor* pActor;

		/// Command objects
		//G4UIcmdWithAnInteger * pNBinsCmd;
		//G4UIcmdWithADoubleAndUnit * pEmaxCmd;

};

#endif
#endif
