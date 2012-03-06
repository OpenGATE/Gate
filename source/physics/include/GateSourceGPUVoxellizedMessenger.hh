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
#include "G4UImessenger.hh"
#include "GateMessenger.hh"
#include "GateSourceVoxellizedMessenger.hh"

class GateSourceGPUVoxellized;

class GateClock;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

#include "GateUIcmdWithAVector.hh"

class GateSourceGPUVoxellizedMessenger: public GateSourceVoxellizedMessenger
{
	public:
		GateSourceGPUVoxellizedMessenger(GateSourceGPUVoxellized* source);
		~GateSourceGPUVoxellizedMessenger();

		void SetNewValue(G4UIcommand*, G4String);

	private:
		GateSourceGPUVoxellized* m_gpu_source;

};

#endif

