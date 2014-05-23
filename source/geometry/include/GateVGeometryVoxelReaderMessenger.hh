/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVGeometryVoxelReaderMessenger_h
#define GateVGeometryVoxelReaderMessenger_h 1

class G4UIdirectory;
class G4UIcommand;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

class GateVGeometryVoxelReader;

#include "GateVGeometryVoxelStoreMessenger.hh"

class GateVGeometryVoxelReaderMessenger : public GateVGeometryVoxelStoreMessenger
{
public:
  GateVGeometryVoxelReaderMessenger(GateVGeometryVoxelReader* voxelReader);
  virtual ~GateVGeometryVoxelReaderMessenger();
  void SetNewValue(G4UIcommand* command, G4String newValue);

protected:

  G4UIcmdWithAString*                    ReadFileCmd;
  G4UIcmdWithAnInteger*                  DescribeCmd;

  G4UIcmdWithAString*                    InsertTranslatorCmd;
  G4UIcmdWithoutParameter*               RemoveTranslatorCmd;

  GateVGeometryVoxelReader*                   m_voxelReader;
};

#endif
