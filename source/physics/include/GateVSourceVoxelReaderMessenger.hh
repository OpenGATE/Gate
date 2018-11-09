/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#ifndef GATEVSOURCEVOXELREADERMESSENGER_H
#define GATEVSOURCEVOXELREADERMESSENGER_H 1

#include "GateMessenger.hh"
#include "GateUIcmdWithAVector.hh"

class GateVSourceVoxelReader;
class GateClock;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

//-----------------------------------------------------------------------------
class GateVSourceVoxelReaderMessenger: public GateMessenger
{
public:
  GateVSourceVoxelReaderMessenger(GateVSourceVoxelReader* voxelReader);
  ~GateVSourceVoxelReaderMessenger();

  void SetNewValue(G4UIcommand* , G4String);

protected:
  GateVSourceVoxelReader*             m_voxelReader;
  G4UIcmdWith3VectorAndUnit*          PositionCmd;
  G4UIcmdWith3VectorAndUnit*          VoxelSizeCmd;
  GateUIcmdWithAVector<G4String>*     InsertTranslatorCmd;
  G4UIcmdWithoutParameter*            RemoveTranslatorCmd;
  G4UIcmdWithAnInteger*               VerboseCmd;
  G4UIcmdWithAString*                 TimeActivTablesCmd;
  G4UIcmdWithAString*                 ActivityImageCmd;
  G4UIcmdWithADoubleAndUnit*          SetTimeSamplingCmd;
};
//-----------------------------------------------------------------------------

#endif
