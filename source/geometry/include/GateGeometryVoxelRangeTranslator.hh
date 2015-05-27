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
#include "GateRangeMaterialTable.hh"
#include "G4VisAttributes.hh"

class GateGeometryVoxelRangeTranslatorMessenger;

class GateGeometryVoxelRangeTranslator : public GateVGeometryVoxelTranslator
{
public:
  GateGeometryVoxelRangeTranslator(GateVGeometryVoxelReader* voxelReader);
  virtual ~GateGeometryVoxelRangeTranslator();
    typedef std::map<G4Material*, G4VisAttributes*>  GateVoxelAttributesTranslationMap;
    typedef GateRangeMaterialTable::iterator iterator;
    typedef GateRangeMaterialTable::const_iterator const_iterator;
    iterator begin(){ return m_voxelMaterialTranslation.begin(); }
    iterator end(){ return m_voxelMaterialTranslation.end(); }
  
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
  
  inline GateVoxelAttributesTranslationMap GetAttributesMap(){ return m_voxelAttributesTranslation; }

  //! Modif DS: Method to iterate through the material's list
  G4String GetNextMaterial(G4bool doReset=false);

  //! Used by GateRegularParameterization to get the different materials
  void GetCompleteListOfMaterials(std::vector<G4String>& mat);
  
  inline void clear(){ m_voxelMaterialTranslation.clear(); }
  inline void push_back(GateRangeMaterialTable::mMaterials MM){ m_voxelMaterialTranslation.push_back(MM); }
  inline void pop_back(){ m_voxelMaterialTranslation.pop_back(); }

protected:

  GateRangeMaterialTable                    m_voxelMaterialTranslation;

  GateVoxelAttributesTranslationMap                            m_voxelAttributesTranslation;

  GateGeometryVoxelRangeTranslatorMessenger*                   m_messenger; 
  
private:
  // Iterator for the GetNextMaterial function
  GateRangeMaterialTable::iterator anIterator;

};

#endif
