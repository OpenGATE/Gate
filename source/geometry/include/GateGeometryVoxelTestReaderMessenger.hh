/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateGeometryVoxelTestReaderMessenger_h
#define GateGeometryVoxelTestReaderMessenger_h 1

class G4UIdirectory;
class G4UIcommand;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

class GateGeometryVoxelTestReader;

#include "GateVGeometryVoxelReaderMessenger.hh"

class GateGeometryVoxelTestReaderMessenger : public GateVGeometryVoxelReaderMessenger
{
public:
  GateGeometryVoxelTestReaderMessenger(GateGeometryVoxelTestReader* voxelReader);
  virtual ~GateGeometryVoxelTestReaderMessenger();

  void SetNewValue(G4UIcommand* command, G4String newValue);

protected:

  GateGeometryVoxelTestReader*                   m_voxelReader;
};

#endif
