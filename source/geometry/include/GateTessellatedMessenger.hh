/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#ifndef GateTessellatedMessenger_h
#define GateTessellatedMessenger_h 1

#include "globals.hh"
#include "GateVolumeMessenger.hh"

class GateTessellated;

class GateTessellatedMessenger : public GateVolumeMessenger
{
  public:
    GateTessellatedMessenger( GateTessellated* itsCreator );
    virtual ~GateTessellatedMessenger();

    void SetNewValue( G4UIcommand*, G4String );
    virtual inline GateTessellated* GetTessellatedCreator()
    {
      return (GateTessellated*)GetVolumeCreator();
    }

  private:
    G4UIcmdWithAString* PathToSTLFileCmd;
};

#endif
