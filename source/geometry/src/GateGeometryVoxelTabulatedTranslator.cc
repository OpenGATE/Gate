/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include <fstream>

#include "GateGeometryVoxelTabulatedTranslator.hh"
#include "GateGeometryVoxelTabulatedTranslatorMessenger.hh"
#include "G4Colour.hh"

#include "GateDetectorConstruction.hh"
#include "GateMaterialDatabase.hh"

//-----------------------------------------------------------------------------------------------
GateGeometryVoxelTabulatedTranslator::GateGeometryVoxelTabulatedTranslator(GateVGeometryVoxelReader* voxelReader) 
  : GateVGeometryVoxelTranslator(voxelReader)
{
  m_name = G4String("tabulatedTranslator");
  m_messenger = new GateGeometryVoxelTabulatedTranslatorMessenger(this);
}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
GateGeometryVoxelTabulatedTranslator::~GateGeometryVoxelTabulatedTranslator() 
{
  delete m_messenger;
}

G4String GateGeometryVoxelTabulatedTranslator::TranslateToMaterial(G4int voxelValue)
{
  G4String material = G4String("NULL");

  if (m_voxelMaterialTranslation.find(voxelValue) != m_voxelMaterialTranslation.end()) {
    material = m_voxelMaterialTranslation[voxelValue];
  }

  return material;
}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
void GateGeometryVoxelTabulatedTranslator::ReadTranslationTable(G4String fileName)
{
  std::ifstream inFile;
  G4cout << "GateGeometryVoxelTabulatedTranslator::ReadFile : fileName: " << fileName << G4endl;
  inFile.open(fileName.c_str(),std::ios::in);

  G4String material;
  G4int imageValue;
  G4int nTotCol;
  G4double red, green, blue, alpha;
  G4bool visible;

  char buffer[200];

  inFile.getline(buffer,200);
  std::istringstream  is(buffer);

  is >> nTotCol;
  G4cout << "nTotCol: " << nTotCol << G4endl;

  for (G4int iCol=0; iCol<nTotCol; iCol++) {
    inFile.getline(buffer,200);
    is.clear();
    is.str(buffer);

    is >> imageValue >> material >> std::ws ;
    if (is.eof()){
      visible=true;
      red=0.5;
      green=blue=0.0;
      alpha=1;
    } else {
      is >> std::boolalpha >> visible >> red >> green >> blue >> alpha ;
    }
    
    G4cout << std::boolalpha << "  imageValue: " << imageValue << "  material: " << material 
	   <<", visible " << visible << ", rgba(" << red << ',' << green << ',' << blue<< ',' << alpha << ')' << G4endl;
    

    m_voxelMaterialTranslation[imageValue] = material;
    m_voxelAttributesTranslation[ GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial(material) ]= new G4VisAttributes(visible,G4Colour(red,green,blue,alpha));

  }

  inFile.close();

}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
void GateGeometryVoxelTabulatedTranslator::Describe(G4int) 
{
}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
G4String GateGeometryVoxelTabulatedTranslator::GetNextMaterial(G4bool doReset)
{
  static GateVoxelMaterialTranslationMap::iterator anIterator = m_voxelMaterialTranslation.begin();
  
  if (doReset)
    anIterator = m_voxelMaterialTranslation.begin();
    
  G4String aMaterial = ( anIterator!=m_voxelMaterialTranslation.end() ) ? anIterator->second : G4String("") ;
  if (aMaterial!="")
    anIterator++;
    
  return aMaterial;
}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
//! Used by GateRegularParameterization to get the different materials
void GateGeometryVoxelTabulatedTranslator::GetCompleteListOfMaterials(std::vector<G4String>& mat)
{
  GateVoxelMaterialTranslationMap::iterator it;
  it = m_voxelMaterialTranslation.begin();

  while(it != m_voxelMaterialTranslation.end()) {
    mat.push_back((*it).second);
    it++;
  }
}
//-----------------------------------------------------------------------------------------------
