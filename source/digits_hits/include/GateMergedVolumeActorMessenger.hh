/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  \class  GateMergedVolumeActorMessenger
  \author Didier Benoit (didier.benoit@inserm.fr)
*/

#ifndef GATEMERGEDVOLUMEACTORMESSENGER
#define GATEMERGEDVOLUMEACTORMESSENGER

#include "GateActorMessenger.hh"

class GateMergedVolumeActor;

class GateMergedVolumeActorMessenger : public GateActorMessenger
{
  public:
    GateMergedVolumeActorMessenger( GateMergedVolumeActor* sensor );
    virtual ~GateMergedVolumeActorMessenger();

    void SetNewValue(G4UIcommand*, G4String);

  private:
    void BuildCommands(G4String base);

  private:
    GateMergedVolumeActor* pMergedVolumeActor;
    G4UIcmdWithAString* ListVolumeToMergeCmd;
};

#endif
