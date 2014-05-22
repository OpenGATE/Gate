/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSourceVoxelRangeTranslatorMessenger_h
#define GateSourceVoxelRangeTranslatorMessenger_h 1

class G4UIdirectory;
class G4UIcommand;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

class GateSourceVoxelRangeTranslator;

#include "GateMessenger.hh"

class GateSourceVoxelRangeTranslatorMessenger : public GateMessenger
{
public:
  GateSourceVoxelRangeTranslatorMessenger(GateSourceVoxelRangeTranslator* voxelTranslator);
  virtual ~GateSourceVoxelRangeTranslatorMessenger();
  void SetNewValue(G4UIcommand* command, G4String newValue);

protected:

  G4UIcmdWithAString*                    ReadTableCmd;
  G4UIcmdWithAnInteger*                  DescribeCmd;

  GateSourceVoxelRangeTranslator* m_voxelTranslator;
};

#endif
