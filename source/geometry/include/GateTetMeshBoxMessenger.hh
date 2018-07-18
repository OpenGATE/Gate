/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
----------------------*/

#ifndef GATE_TET_MESH_BOX_MESSENGER_HH
#define GATE_TET_MESH_BOX_MESSENGER_HH

#include <G4String.hh>
#include <G4UIcommand.hh>
#include <G4UIcmdWithAString.hh>
#include <G4UIcmdWithADoubleAndUnit.hh>
#include <G4UIcmdWith3VectorAndUnit.hh>

#include "GateVolumeMessenger.hh"

class GateTetMeshBox;


class GateTetMeshBoxMessenger : public GateVolumeMessenger
{
  public:
    explicit GateTetMeshBoxMessenger(GateTetMeshBox* itsCreator);
    ~GateTetMeshBoxMessenger() final;

    void SetNewValue(G4UIcommand*, G4String) final;

  private:
    G4UIcmdWithAString* pSetPathToAttributeMapCmd;
    G4UIcmdWithAString* pSetPathToELEFileCmd;
    G4UIcmdWithADoubleAndUnit* pSetUnitOfLengthCmd;
};

#endif  // GATE_TET_MESH_BOX_MESSENGER_HH
