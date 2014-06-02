/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
#ifndef GATEGPUEMISTOMOMESSENGER_H
#define GATEGPUEMISTOMOMESSENGER_H 1

#include "globals.hh"
#include "GateSourceVoxellizedMessenger.hh"
#include "G4UIcmdWithAString.hh"

class GateGPUEmisTomo;

class GateGPUEmisTomoMessenger: public GateSourceVoxellizedMessenger
{
public:
  GateGPUEmisTomoMessenger(GateGPUEmisTomo* source);
  ~GateGPUEmisTomoMessenger();
  
  virtual void SetNewValue(G4UIcommand*, G4String);
  
private:
  GateGPUEmisTomo * m_gpu_source;
  G4UIcmdWithAString * m_attach_to_cmd;
  G4UIcmdWithAnInteger * m_gpu_buffer_size_cmd;
  G4UIcmdWithAnInteger * m_gpu_device_id_cmd;
};

#endif

