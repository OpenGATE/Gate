/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSourceVoxelTestReaderMessenger_h
#define GateSourceVoxelTestReaderMessenger_h 1

#include "GateVSourceVoxelReaderMessenger.hh"
#include "GateSourceVoxelTestReader.hh"

class GateSourceVoxelTestReaderMessenger : public GateVSourceVoxelReaderMessenger
{
public:
  GateSourceVoxelTestReaderMessenger(GateSourceVoxelTestReader* voxelReader);
  virtual ~GateSourceVoxelTestReaderMessenger();
  void SetNewValue(G4UIcommand* command, G4String newValue);

protected:
  GateUIcmdWithAVector<G4String>*     ReadFileCmd;

};

#endif
