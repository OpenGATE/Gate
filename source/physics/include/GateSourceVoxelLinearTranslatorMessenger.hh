/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSourceVoxelLinearTranslatorMessenger_h
#define GateSourceVoxelLinearTranslatorMessenger_h 1

class G4UIdirectory;
class G4UIcommand;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

class GateSourceVoxelLinearTranslator;

#include "GateMessenger.hh"

class GateSourceVoxelLinearTranslatorMessenger : public GateMessenger
{
public:
  GateSourceVoxelLinearTranslatorMessenger(GateSourceVoxelLinearTranslator* voxelTranslator);
  virtual ~GateSourceVoxelLinearTranslatorMessenger();
  void SetNewValue(G4UIcommand* command, G4String newValue);

protected:
  G4UIcmdWithADoubleAndUnit*     ValueToActivityScaleCmd;

  GateSourceVoxelLinearTranslator* m_voxelTranslator;
};

#endif
