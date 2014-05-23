/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateGeometryVoxelInterfileReaderMessenger_h
#define GateGeometryVoxelInterfileReaderMessenger_h 1

class G4UIdirectory;
class G4UIcommand;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

class GateGeometryVoxelInterfileReader;

#include "GateVGeometryVoxelReaderMessenger.hh"

class GateGeometryVoxelInterfileReaderMessenger : public GateVGeometryVoxelReaderMessenger
{
public:
  GateGeometryVoxelInterfileReaderMessenger(GateGeometryVoxelInterfileReader* voxelReader);
  virtual ~GateGeometryVoxelInterfileReaderMessenger();

  void SetNewValue(G4UIcommand* command, G4String newValue);

protected:

  GateGeometryVoxelInterfileReader*                   m_voxelReader;
};

#endif
