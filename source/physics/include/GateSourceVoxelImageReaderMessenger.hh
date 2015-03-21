/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATESOURCEVOXELIMAGEREADERMESSENGER_H
#define GATESOURCEVOXELIMAGEREADERMESSENGER_H 1

#include "GateVSourceVoxelReaderMessenger.hh"
#include "GateSourceVoxelImageReader.hh"

//-----------------------------------------------------------------------------
class GateSourceVoxelImageReaderMessenger : public GateVSourceVoxelReaderMessenger
{
public:
  GateSourceVoxelImageReaderMessenger(GateSourceVoxelImageReader* voxelReader);
  virtual ~GateSourceVoxelImageReaderMessenger();

  void SetNewValue(G4UIcommand* command,G4String newValue);

protected:
  GateUIcmdWithAVector<G4String>*     ReadFileCmd;

};
//-----------------------------------------------------------------------------

#endif
