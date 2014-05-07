/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVoxelBoxParameterizedMessenger_h
#define GateVoxelBoxParameterizedMessenger_h 1

#include "globals.hh"

#include "GateMessenger.hh"

class GateVoxelBoxParameterized;

class GateVoxelBoxParameterizedMessenger: public GateMessenger 
{
  public:
    GateVoxelBoxParameterizedMessenger(GateVoxelBoxParameterized* itsInserter);
   ~GateVoxelBoxParameterizedMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);
    
    virtual inline GateVoxelBoxParameterized* GetVoxelParameterizedInserter() 
      { return m_inserter; }

  private:

    G4UIcmdWithoutParameter*        AttachPhantomSDCmd;
    G4UIcmdWithAString*             InsertReaderCmd;
    G4UIcmdWithoutParameter*        RemoveReaderCmd;
    G4UIcmdWithAString*             AddOutputCmd;

    GateVoxelBoxParameterized*  m_inserter;
  
};

#endif

