/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVSourceVoxelTranslator_h
#define GateVSourceVoxelTranslator_h 1

#include "GateVSourceVoxelReader.hh"
#include "globals.hh"

class GateVSourceVoxelTranslator
{
public:
  GateVSourceVoxelTranslator(GateVSourceVoxelReader* voxelReader);
  virtual ~GateVSourceVoxelTranslator() {};
  
public:

  virtual G4double TranslateToActivity(G4int voxelValue) = 0;

  virtual GateVSourceVoxelReader* GetReader() { return m_voxelReader; };
  virtual G4String                GetName()   { return m_name; };

  virtual void UpdateActivity(G4double , G4double  , G4double ) = 0; /* PY Descourt 08/09/2009 */
  virtual void Describe(G4int) = 0;
protected:
  G4String                       m_name;
  GateVSourceVoxelReader*        m_voxelReader;

};

#endif

