/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateVoxelizedMassActorMessenger
  \author Thomas DESCHLER (thomas.deschler@iphc.cnrs.fr)
*/

#ifndef GATEVOXELIZEDMASSACTORMESSENGER_HH
#define GATEVOXELIZEDMASSACTORMESSENGER_HH

#include "G4UIcmdWithABool.hh"
#include "GateImageActorMessenger.hh"

class GateVoxelizedMassActor;
class GateVoxelizedMassActorMessenger : public GateImageActorMessenger
{
public:

  GateVoxelizedMassActorMessenger(GateVoxelizedMassActor* sensor);
  virtual ~GateVoxelizedMassActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:

  GateVoxelizedMassActor* pVoxelizedMassActor;

  G4UIcmdWithABool* pEnableMassCmd;
};

#endif
