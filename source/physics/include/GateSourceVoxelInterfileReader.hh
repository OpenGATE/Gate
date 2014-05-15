/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSourceVoxelInterfileReader_h
#define GateSourceVoxelInterfileReader_h 1

#include "GateVSourceVoxelReader.hh"
class GateSourceVoxelInterfileReaderMessenger;

class GateSourceVoxelInterfileReader : public GateVSourceVoxelReader
{
public:
  GateSourceVoxelInterfileReader(GateVSource* source);
  
  virtual ~GateSourceVoxelInterfileReader();

  void ReadFile(G4String fileName);

  void ReadKey(FILE* fp);
  
  /* PY Descourt 08/09/2009 */
  void ReadRTFile(G4String, G4String);
  void ReadKeyFrames(FILE*) ;
  /* PY Descourt 08/09/2009 */
  
protected:
  
  typedef unsigned short G4short;
  
  inline void SwapBytes(G4short* buffer, G4int size) {
  	for (G4int i = 0; i < size; ++i) {
  		buffer[i] = ((buffer[i]>> 8) | (buffer[i] << 8));
  	}
  }
  
  G4String m_dataFileName;
  G4int m_numPlanes;
  G4float m_planeThickness;
  G4int m_dim[2];
  G4float m_pixelSize[2];
  G4float m_matrixSize[2];
  G4String m_dataTypeName;
  G4String m_dataByteOrder; // Added by HDS : For BIG/LITTLE ENDIAN interfile support
  GateSourceVoxelInterfileReaderMessenger* m_messenger;
  G4bool IsFirstFrame; // for RTPhantom/* PY Descourt 08/09/2009 */
};

#endif
