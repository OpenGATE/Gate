/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVoxelCompressorMessenger_h
#define GateVoxelCompressorMessenger_h 1

#include "globals.hh"

#include "GateMessenger.hh"

class GateVoxelCompressor;

class GateVoxelCompressorMessenger: public GateMessenger 
{
  public:
    GateVoxelCompressorMessenger(GateVoxelCompressor* itsInserter);
   ~GateVoxelCompressorMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);

  private:

    G4UIcmdWithAString*             MakeExclusionListCmd;
    GateVoxelCompressor*            m_inserter;
  
};

#endif

