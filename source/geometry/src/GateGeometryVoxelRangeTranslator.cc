/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateGeometryVoxelRangeTranslator.hh"
#include "GateGeometryVoxelRangeTranslatorMessenger.hh"

#include "GateDetectorConstruction.hh"
#include "GateMaterialDatabase.hh"

#include "G4ios.hh"
#include <iomanip>
#include <fstream>

#include "G4Colour.hh"


GateGeometryVoxelRangeTranslator::GateGeometryVoxelRangeTranslator(GateVGeometryVoxelReader* voxelReader) 
  : GateVGeometryVoxelTranslator(voxelReader)
{
//  G4cout << " Constructor GateGeometryVoxelRangeTranslator\n";
  m_name = G4String("rangeTranslator");
  anIterator = m_voxelMaterialTranslation.begin();
  m_messenger = new GateGeometryVoxelRangeTranslatorMessenger(this);
  
//  G4cout << " FIN Constructor GateGeometryVoxelRangeTranslator\n";
}

GateGeometryVoxelRangeTranslator::~GateGeometryVoxelRangeTranslator() 
{
  delete m_messenger;
}

G4String GateGeometryVoxelRangeTranslator::TranslateToMaterial(G4double voxelValue)
{
  G4String material = G4String("NULL");

//    G4cout << "GateGeometryVoxelRangeTranslator::TranslateToMaterial: voxelValue " << voxelValue << Gateendl;
  GateVoxelMaterialTranslationRangeVector::iterator itr;

  for (itr = m_voxelMaterialTranslation.begin(); itr != m_voxelMaterialTranslation.end(); itr++) {
    G4double range1 = (itr->first).first;
    G4double range2 = (itr->first).second;
    //    G4cout << "iRange range1 range2 " << iRange << " " << range1 << " " << range2 << Gateendl;
    if ((range1 <= voxelValue) && (voxelValue <= range2)) {
      material = (itr->second);
      break;
    }
  }

  return material;
}

void GateGeometryVoxelRangeTranslator::ReadTranslationTable(G4String fileName)
{

//  G4cout << " DEBUT GateGeometryVoxelRangeTranslator::ReadTranslationTable \n";
  m_voxelMaterialTranslation.clear();

  std::ifstream inFile;
  
//  G4cout << "GateGeometryVoxelRangeTranslator::ReadFile : fileName: " << fileName << Gateendl;
  
  inFile.open(fileName.c_str(),std::ios::in);
   
  if (inFile.is_open()){
  G4String material;
  G4double xmin;
  G4double xmax;
  G4int nTotCol;

  G4double red, green, blue, alpha;
  G4bool visible;
  char buffer [200];

  inFile.getline(buffer,200);
  std::istringstream is(buffer);

  is >> nTotCol;
  //  G4cout << "nTotCol: " << nTotCol << Gateendl;

  for (G4int iCol=0; iCol<nTotCol; iCol++) {
    inFile.getline(buffer,200);
    is.clear();
    is.str(buffer);

    is >> xmin >> xmax;
    is >> material;

    if (is.eof()){
      visible=true;
      red=0.5;
      green=blue=0.0;
      alpha=1;
    }else{
      is >> std::boolalpha >> visible >> red >> green >> blue >> alpha;
    }

    //   G4cout << " min max " << xmin << " " << xmax << "  material: " << material 
    //   << std::boolalpha << ", visible " << visible << ", rgba(" << red<<',' << green << ',' << blue << ')' << Gateendl;

    std::pair<G4double,G4double> minmax(xmin, xmax);
    GateVoxelMaterialTranslationRange range(minmax, material);

    // Add check on possible overlaps with previously defined image value ranges
    // before adding this range to the range table

    m_voxelMaterialTranslation.push_back(range);

/*old class GateMaterialDatabase
    m_voxelAttributesTranslation[  GateMaterialDatabase::GetInstance()->GetMaterial(material) ] =
      new G4VisAttributes(visible, G4Colour(red, green, blue, alpha));
*/    
    m_voxelAttributesTranslation[theMaterialDatabase.GetMaterial(material) ] =
      new G4VisAttributes(visible, G4Colour(red, green, blue, alpha));
  }

  }
  else {G4cout << "Error opening file.\n";}
  
  inFile.close();

}

void GateGeometryVoxelRangeTranslator::Describe(G4int) 
{
  G4cout << " Range Translator\n";
  for (G4int iRange = 0; iRange< (G4int)m_voxelMaterialTranslation.size(); iRange++) {
    G4double    xmin      = (m_voxelMaterialTranslation[iRange].first).first;
    G4double    xmax      = (m_voxelMaterialTranslation[iRange].first).second;
    G4String material = (m_voxelMaterialTranslation[iRange].second);
    G4cout << "\tRange "  << std::setw(3) << iRange 
	   << " : imageValue in [ " 
	   << std::resetiosflags(std::ios::floatfield) 
	   << std::setiosflags(std::ios::scientific) 
	   << std::setprecision(3) 
	   << std::setw(12) 
	   << xmin 
	   << " , "   
	   << xmax 
	   << " ]  ---> material " << material 
	   << ", visibility " << GetMaterialAttributes( theMaterialDatabase.GetMaterial(material) )->IsVisible()
	   << ", coulour "    << GetMaterialAttributes( theMaterialDatabase.GetMaterial(material) )->GetColour()
	   << Gateendl;
  }
}

G4String GateGeometryVoxelRangeTranslator::GetNextMaterial(G4bool doReset)
{
  //static GateVoxelMaterialTranslationRangeVector::iterator anIterator = m_voxelMaterialTranslation.begin();
  
  if (doReset)
    anIterator = m_voxelMaterialTranslation.begin();
    
  G4String aMaterial = ( anIterator!=m_voxelMaterialTranslation.end() ) ? anIterator->second : G4String("") ;
  if (aMaterial!="")
    anIterator++;
    
  return aMaterial;
}

//! Used by GateRegularParameterization to get the different materials
void GateGeometryVoxelRangeTranslator::GetCompleteListOfMaterials(std::vector<G4String>& mat)
{
  GateVoxelMaterialTranslationRangeVector::iterator itr;

  for (itr = m_voxelMaterialTranslation.begin(); itr != m_voxelMaterialTranslation.end(); itr++) {
    mat.push_back(itr->second);
  }
}
