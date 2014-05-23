/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/



#ifndef GateVGeometryVoxelStoreMessenger_h
#define GateVGeometryVoxelStoreMessenger_h 1

class G4UIdirectory;
class G4UIcommand;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

class GateVGeometryVoxelStore;

#include "GateMessenger.hh"

class GateVGeometryVoxelStoreMessenger : public GateMessenger
{
public:
  GateVGeometryVoxelStoreMessenger(GateVGeometryVoxelStore* voxelStore);
  virtual ~GateVGeometryVoxelStoreMessenger();
  void SetNewValue(G4UIcommand* command, G4String newValue);

protected:

  G4UIcmdWithAString*                    DefaultMaterialCmd;

  GateVGeometryVoxelStore*               m_voxelStore;
};

#endif
