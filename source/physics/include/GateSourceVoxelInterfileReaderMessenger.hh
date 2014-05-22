/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSourceVoxelInterfileReaderMessenger_h
#define GateSourceVoxelInterfileReaderMessenger_h 1

#include "GateVSourceVoxelReaderMessenger.hh"
#include "GateSourceVoxelInterfileReader.hh"

class GateSourceVoxelInterfileReaderMessenger : public GateVSourceVoxelReaderMessenger
{
public:
  GateSourceVoxelInterfileReaderMessenger(GateSourceVoxelInterfileReader* voxelReader);
  virtual ~GateSourceVoxelInterfileReaderMessenger();

  void SetNewValue(G4UIcommand* command,G4String newValue);

protected:
  GateUIcmdWithAVector<G4String>*     ReadFileCmd;

};

#endif

