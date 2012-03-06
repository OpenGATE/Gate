/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateSourceGPUVoxellizedMessenger.hh"
#include "GateSourceGPUVoxellized.hh"

#include "GateClock.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

GateSourceGPUVoxellizedMessenger::GateSourceGPUVoxellizedMessenger(GateSourceGPUVoxellized* source)
: GateSourceVoxellizedMessenger(source), m_gpu_source(source)
{ 
}

GateSourceGPUVoxellizedMessenger::~GateSourceGPUVoxellizedMessenger()
{
}

void GateSourceGPUVoxellizedMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
	G4cout << " GateSourceGPUVoxellizedMessenger::SetNewValue" << G4endl;
	GateSourceVoxellizedMessenger::SetNewValue(command,newValue);
}

