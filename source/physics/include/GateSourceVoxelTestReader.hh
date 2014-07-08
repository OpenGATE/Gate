/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSourceVoxelTestReader_h
#define GateSourceVoxelTestReader_h 1

#include "GateVSourceVoxelReader.hh"
class GateSourceVoxelTestReaderMessenger;

class GateSourceVoxelTestReader : public GateVSourceVoxelReader
{
public:
  GateSourceVoxelTestReader(GateVSource* source);
  virtual ~GateSourceVoxelTestReader();

  void ReadFile(G4String fileName);
  void ReadRTFile(G4String, G4String);/* PY Descourt 08/09/2009 */
protected:

  GateSourceVoxelTestReaderMessenger* m_messenger;

};

#endif
