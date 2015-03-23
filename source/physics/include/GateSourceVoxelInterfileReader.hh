/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSourceVoxelInterfileReader_h
#define GateSourceVoxelInterfileReader_h 1

#include "GateVSourceVoxelReader.hh"
#include "GateInterfileHeader.hh"
class GateSourceVoxelInterfileReaderMessenger;

class GateSourceVoxelInterfileReader : public GateVSourceVoxelReader, public GateInterfileHeader
{
public:
  GateSourceVoxelInterfileReader(GateVSource* source);
  
  virtual ~GateSourceVoxelInterfileReader();

  void ReadFile(G4String fileName);
  
  /* PY Descourt 08/09/2009 */
  void ReadRTFile(G4String, G4String);
  /* PY Descourt 08/09/2009 */
  
protected:
  GateSourceVoxelInterfileReaderMessenger* m_messenger;
  G4bool IsFirstFrame; // for RTPhantom/* PY Descourt 08/09/2009 */
};

#endif
