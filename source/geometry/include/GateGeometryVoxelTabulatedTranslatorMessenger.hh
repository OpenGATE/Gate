/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateGeometryVoxelTabulatedTranslatorMessenger_h
#define GateGeometryVoxelTabulatedTranslatorMessenger_h 1

class G4UIdirectory;
class G4UIcommand;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

class GateGeometryVoxelTabulatedTranslator;

#include "GateMessenger.hh"

class GateGeometryVoxelTabulatedTranslatorMessenger : public GateMessenger
{
public:
  GateGeometryVoxelTabulatedTranslatorMessenger(GateGeometryVoxelTabulatedTranslator* voxelTranslator);
  virtual ~GateGeometryVoxelTabulatedTranslatorMessenger();
  void SetNewValue(G4UIcommand* command, G4String newValue);

protected:

  G4UIcmdWithAString*                    ReadTableCmd;
  G4UIcmdWithAnInteger*                  DescribeCmd;

  GateGeometryVoxelTabulatedTranslator* m_voxelTranslator;
};

#endif
