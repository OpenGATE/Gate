/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSourceVoxelRangeTranslator_h
#define GateSourceVoxelRangeTranslator_h 1

#include "globals.hh"
#include "GateVSourceVoxelTranslator.hh"

class GateSourceVoxelRangeTranslatorMessenger;

class GateSourceVoxelRangeTranslator : public GateVSourceVoxelTranslator
{
public:
  GateSourceVoxelRangeTranslator(GateVSourceVoxelReader* voxelReader);
  virtual ~GateSourceVoxelRangeTranslator();
  
  /* PY Descourt 08/09/2009 */
  void     ReadTranslationTable(G4String fileName);
  void     AddTranslationRange( G4double rmin , G4double rmax ) ;
  void     Describe(G4int level);

public:

  G4double TranslateToActivity(G4int voxelValue) {
    G4double xVoxelValue = voxelValue; 
    return TranslateToActivity(xVoxelValue);
  };
  G4double TranslateToActivity(G4double voxelValue);
  void UpdateActivity(G4double, G4double, G4double);
protected:

  typedef std::pair<std::pair<G4double,G4double>,G4double>   GateVoxelActivityTranslationRange;
  typedef std::vector<GateVoxelActivityTranslationRange> GateVoxelActivityTranslationRangeVector;
  GateVoxelActivityTranslationRangeVector    m_voxelActivityTranslation;

  GateSourceVoxelRangeTranslatorMessenger* m_messenger; 

};

#endif
