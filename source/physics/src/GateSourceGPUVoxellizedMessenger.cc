/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateSourceGPUVoxellizedMessenger.hh"
#include "GateSourceGPUVoxellized.hh"

GateSourceGPUVoxellizedMessenger::GateSourceGPUVoxellizedMessenger(GateSourceGPUVoxellized* source)
: GateSourceVoxellizedMessenger(source), m_gpu_source(source)
{ 
	m_attach_to_cmd = new G4UIcmdWithAString((GetDirectoryName()+"attachTo").c_str(),this);
	m_attach_to_cmd->SetGuidance("Attach to gate volume");

}

GateSourceGPUVoxellizedMessenger::~GateSourceGPUVoxellizedMessenger()
{
}

void GateSourceGPUVoxellizedMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
	G4cout << " GateSourceGPUVoxellizedMessenger::SetNewValue" << G4endl;
	if (command == m_attach_to_cmd) m_gpu_source->AttachToVolume(newValue);
	GateSourceVoxellizedMessenger::SetNewValue(command,newValue);
}

