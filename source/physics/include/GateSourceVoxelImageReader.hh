/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSourceVoxelImageReader_h
#define GateSourceVoxelImageReader_h 1

#include "GateVSourceVoxelReader.hh"
class GateSourceVoxelImageReaderMessenger;

class GateSourceVoxelImageReader : public GateVSourceVoxelReader
{
public:
  GateSourceVoxelImageReader(GateVSource* source);
  virtual ~GateSourceVoxelImageReader();

  void ReadFile(G4String fileName);
  void ReadRTFile(G4String, G4String); /* PY Descourt 08/09/2009 */
protected:

  GateSourceVoxelImageReaderMessenger* m_messenger;
};

#endif
