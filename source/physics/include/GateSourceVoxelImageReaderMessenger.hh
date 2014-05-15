/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSourceVoxelImageReaderMessenger_h
#define GateSourceVoxelImageReaderMessenger_h 1

#include "GateVSourceVoxelReaderMessenger.hh"
#include "GateSourceVoxelImageReader.hh"

class GateSourceVoxelImageReaderMessenger : public GateVSourceVoxelReaderMessenger
{
public:
  GateSourceVoxelImageReaderMessenger(GateSourceVoxelImageReader* voxelReader);
  virtual ~GateSourceVoxelImageReaderMessenger();

  void SetNewValue(G4UIcommand* command,G4String newValue);

protected:
  GateUIcmdWithAVector<G4String>*     ReadFileCmd;

};

#endif

