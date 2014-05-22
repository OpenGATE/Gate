/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSourceVoxelLinearTranslator_h
#define GateSourceVoxelLinearTranslator_h 1

#include "globals.hh"
#include "GateVSourceVoxelTranslator.hh"

class GateSourceVoxelLinearTranslatorMessenger;

class GateSourceVoxelLinearTranslator : public GateVSourceVoxelTranslator
{
public:
  GateSourceVoxelLinearTranslator(GateVSourceVoxelReader* voxelReader);
  virtual ~GateSourceVoxelLinearTranslator();
  
  void     SetValueToActivityScale(G4double value) { m_valueToActivityScale = value; };
  G4double GetValueToActivityScale()               { return m_valueToActivityScale; };
  void Describe(G4int){ };
public:

  G4double TranslateToActivity(G4int voxelValue);
  void UpdateActivity(G4double , G4double , G4double); // PY Descourt 08/09/2009
protected:

  G4double m_valueToActivityScale;

  GateSourceVoxelLinearTranslatorMessenger* m_messenger; 

};

#endif
