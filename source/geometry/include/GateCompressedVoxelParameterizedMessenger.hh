/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCompressedVoxelParameterizedMessenger_h
#define GateCompressedVoxelParameterizedMessenger_h 1

#include "globals.hh"

#include "GateMessenger.hh"

class GateCompressedVoxelParameterized;

class GateCompressedVoxelParameterizedMessenger: public GateMessenger 
{
  public:
    GateCompressedVoxelParameterizedMessenger(GateCompressedVoxelParameterized* itsInserter);
   ~GateCompressedVoxelParameterizedMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);
    
    virtual inline GateCompressedVoxelParameterized* GetVoxelParameterizedInserter() 
      { return m_inserter; }

  private:

    G4UIcmdWithoutParameter*        AttachPhantomSDCmd;
    G4UIcmdWithAString*             InsertReaderCmd;
    G4UIcmdWithoutParameter*        RemoveReaderCmd;
    G4UIcmdWithAString*             AddOutputCmd;

    GateCompressedVoxelParameterized*  m_inserter;
  
};

#endif

