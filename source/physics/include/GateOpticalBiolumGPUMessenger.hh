/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
#ifndef GATEOPTICALBIOLUMGPUMESSENGER_H
#define GATEOPTICALBIOLUMGPUMESSENGER_H 1

#include "globals.hh"
#include "GateSourceVoxellizedMessenger.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

class GateOpticalBiolumGPU;

class GateOpticalBiolumGPUMessenger: public GateSourceVoxellizedMessenger
{
public:
  GateOpticalBiolumGPUMessenger(GateOpticalBiolumGPU* source);
  ~GateOpticalBiolumGPUMessenger();
  
  virtual void SetNewValue(G4UIcommand*, G4String);
  
private:
  GateOpticalBiolumGPU * m_gpu_source;
  G4UIcmdWithAString * m_attach_to_cmd;
  G4UIcmdWithAnInteger * m_gpu_buffer_size_cmd;
  G4UIcmdWithAnInteger * m_gpu_device_id_cmd;
  G4UIcmdWithADoubleAndUnit * m_gpu_energy_cmd;
};

#endif

