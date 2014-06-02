/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateGeometryVoxelRangeTranslator_h
#define GateGeometryVoxelRangeTranslator_h 1

#include "globals.hh"
#include "GateVGeometryVoxelTranslator.hh"
#include "G4VisAttributes.hh"

class GateGeometryVoxelRangeTranslatorMessenger;

class GateGeometryVoxelRangeTranslator : public GateVGeometryVoxelTranslator
{
public:
  GateGeometryVoxelRangeTranslator(GateVGeometryVoxelReader* voxelReader);
  virtual ~GateGeometryVoxelRangeTranslator();
  
  void     ReadTranslationTable(G4String fileName);

  void     Describe(G4int level);

public:

  inline G4VisAttributes* GetMaterialAttributes(G4Material* m){
    return m_voxelAttributesTranslation[m];
  }

  G4String TranslateToMaterial(G4int voxelValue) {
    G4double xVoxelValue = voxelValue; 
    return TranslateToMaterial(xVoxelValue);
  };
  G4String TranslateToMaterial(G4double voxelValue);

  //! Modif DS: Method to iterate through the material's list
  G4String GetNextMaterial(G4bool doReset=false);

  //! Used by GateRegularParameterization to get the different materials
  void GetCompleteListOfMaterials(std::vector<G4String>& mat);

protected:

  typedef std::pair<std::pair<G4double,G4double>,G4String> GateVoxelMaterialTranslationRange;
  typedef std::vector<GateVoxelMaterialTranslationRange>     GateVoxelMaterialTranslationRangeVector;
  GateVoxelMaterialTranslationRangeVector                      m_voxelMaterialTranslation;

  typedef std::map<G4Material*, G4VisAttributes*>            GateVoxelAttributesTranslationMap;
  GateVoxelAttributesTranslationMap                            m_voxelAttributesTranslation;

  GateGeometryVoxelRangeTranslatorMessenger*                   m_messenger; 

};

#endif
