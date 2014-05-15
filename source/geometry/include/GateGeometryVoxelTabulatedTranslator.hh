/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateGeometryVoxelTabulatedTranslator_h
#define GateGeometryVoxelTabulatedTranslator_h 1

#include "globals.hh"
#include "GateVGeometryVoxelTranslator.hh"
#include "G4VisAttributes.hh"

class GateGeometryVoxelTabulatedTranslatorMessenger;

class GateGeometryVoxelTabulatedTranslator : public GateVGeometryVoxelTranslator
{
public:
  GateGeometryVoxelTabulatedTranslator(GateVGeometryVoxelReader* voxelReader);
  virtual ~GateGeometryVoxelTabulatedTranslator();
  
  void     ReadTranslationTable(G4String fileName);

  void     Describe(G4int level);

public:


  inline G4VisAttributes* GetMaterialAttributes(G4Material* m){
    return m_voxelAttributesTranslation[m];
  }

  G4String TranslateToMaterial(G4int voxelValue);

  //! Modif DS: Method to iterate through the material's list
  G4String GetNextMaterial(G4bool doReset=false);

  //! Used by GateRegularParameterization to get the different materials
  void GetCompleteListOfMaterials(std::vector<G4String>& mat);

protected:

  typedef std::map<G4int,G4String> GateVoxelMaterialTranslationMap;
  GateVoxelMaterialTranslationMap    m_voxelMaterialTranslation;
 
 typedef std::map<G4Material*, G4VisAttributes*> GateVoxelAttributesTranslationMap;
  GateVoxelAttributesTranslationMap m_voxelAttributesTranslation;

  GateGeometryVoxelTabulatedTranslatorMessenger* m_messenger; 

};

#endif
