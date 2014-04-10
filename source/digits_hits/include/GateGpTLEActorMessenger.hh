#ifndef GATEGPTLEACTORMESSENGER_HH
#define GATEGPTLEACTORMESSENGER_HH

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

#include "GateActorMessenger.hh"

#include <G4UIcmdWithAnInteger.hh>
#include <G4UIcmdWithADoubleAndUnit.hh>

class GateGpTLEActor;

class GateGpTLEActorMessenger : public GateActorMessenger
{
	public: 

                GateGpTLEActorMessenger(GateGpTLEActor*);
                ~GateGpTLEActorMessenger();

		void SetNewValue(G4UIcommand*, G4String);
	protected:
		void BuildCommands(G4String base);
                GateGpTLEActor* pGpTLEActor;
                G4UIcmdWithAString* pFileSpectreBaseNameCmd;
                G4UIcmdWithAString* pSaveFilenameCmd;
		/// Command objects
		//G4UIcmdWithAnInteger * pNBinsCmd;
		//G4UIcmdWithADoubleAndUnit * pEmaxCmd;

    };

#endif
#endif

