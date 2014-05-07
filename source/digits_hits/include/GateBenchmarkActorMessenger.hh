/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

#ifndef GATEBENCHMARKACTORMESSENGER_HH
#define GATEBENCHMARKACTORMESSENGER_HH

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

#include "GateActorMessenger.hh"

class GateBenchmarkActor;

class GateBenchmarkActorMessenger : public GateActorMessenger
{
	public:

		GateBenchmarkActorMessenger(GateBenchmarkActor * v);

		~GateBenchmarkActorMessenger();

		virtual void SetNewValue(G4UIcommand*, G4String);
	protected:
		void BuildCommands(G4String base);

		GateBenchmarkActor * pActor;

		/// Command objects
		//G4UIcmdWithAnInteger * pNBinsCmd;
		//G4UIcmdWithADoubleAndUnit * pEmaxCmd;

};

#endif
#endif
