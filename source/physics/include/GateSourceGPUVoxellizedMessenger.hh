/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
#ifndef GATESOURCEGPUVOXELLIZEDMESSENGER_H
#define GATESOURCEGPUVOXELLIZEDMESSENGER_H 1

#include "globals.hh"
#include "GateSourceVoxellizedMessenger.hh"
#include "G4UIcmdWithAString.hh"

class GateSourceGPUVoxellized;

class GateSourceGPUVoxellizedMessenger: public GateSourceVoxellizedMessenger
{
	public:
		GateSourceGPUVoxellizedMessenger(GateSourceGPUVoxellized* source);
		~GateSourceGPUVoxellizedMessenger();

		virtual void SetNewValue(G4UIcommand*, G4String);

	private:
		GateSourceGPUVoxellized* m_gpu_source;
		G4UIcmdWithAString* m_attach_to_cmd;

};

#endif

