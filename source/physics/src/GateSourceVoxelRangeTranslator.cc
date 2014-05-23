/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSourceVoxelRangeTranslator.hh"
#include "GateSourceVoxelRangeTranslatorMessenger.hh"

#include "G4SystemOfUnits.hh"
#include "G4ios.hh"
#include <fstream>
#include <iomanip>

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

GateSourceVoxelRangeTranslator::GateSourceVoxelRangeTranslator(GateVSourceVoxelReader* voxelReader) 
  : GateVSourceVoxelTranslator(voxelReader)
{
  m_name = G4String("rangeTranslator");
  m_messenger = new GateSourceVoxelRangeTranslatorMessenger(this);
}

GateSourceVoxelRangeTranslator::~GateSourceVoxelRangeTranslator() 
{
  delete m_messenger;
}

G4double GateSourceVoxelRangeTranslator::TranslateToActivity(G4double voxelValue)
{
  G4double activity = 0.;

  for (G4int iRange = 0; iRange< (G4int)m_voxelActivityTranslation.size(); iRange++) {
    G4double range1 = (m_voxelActivityTranslation[iRange].first).first;
    G4double range2 = (m_voxelActivityTranslation[iRange].first).second;
    //    G4cout << "iRange range1 range2 " << iRange << " " << range1 << " " << range2 << G4endl;
    if ((range1 <= voxelValue) && (voxelValue <= range2)) {
      activity = (m_voxelActivityTranslation[iRange].second);
      break;
    }
  }

  return activity;
}

void GateSourceVoxelRangeTranslator::ReadTranslationTable(G4String fileName)
{
  m_voxelActivityTranslation.clear();

  std::ifstream inFile;
  //  G4cout << "GateSourceVoxelRangeTranslator::ReadFile : fileName: " << fileName << G4endl;
  inFile.open(fileName.c_str(),std::ios::in);

  G4double activity;
  G4double xmin;
  G4double xmax;
  G4int nTotCol;

  inFile >> nTotCol;
  //  G4cout << "nTotCol: " << nTotCol << G4endl;

  for (G4int iCol=0; iCol<nTotCol; iCol++) {

    inFile >> xmin >> xmax;
    inFile >> activity;
    //    G4cout << " min max " << min << " " << max << "  activity: " << activity << G4endl;

    std::pair<G4double,G4double> minmax(xmin, xmax);
    GateVoxelActivityTranslationRange range(minmax, activity * becquerel);

    // Add check on possible overlaps with previously defined image value ranges
    // before adding this range to the range table

    m_voxelActivityTranslation.push_back(range);

  }

  inFile.close();

}

void GateSourceVoxelRangeTranslator::Describe(G4int) 
{
  G4cout << " Range Translator" << G4endl;
  for (G4int iRange = 0; iRange< (G4int)m_voxelActivityTranslation.size(); iRange++) {
    G4double    xmin      = (m_voxelActivityTranslation[iRange].first).first;
    G4double    xmax      = (m_voxelActivityTranslation[iRange].first).second;
    G4double activity = (m_voxelActivityTranslation[iRange].second);
    G4cout << "\tRange "  << std::setw(3) << iRange 
	   << " : imageValue in [ " 
	   << std::resetiosflags(std::ios::floatfield) 
	   << std::setiosflags(std::ios::scientific) 
	   << std::setprecision(3) 
	   << std::setw(12) 
	   << xmin 
	   << " , "   << xmax 
	   << " ]  ---> activity (Bq) " 
	   << activity/becquerel 
	   << G4endl;
  }
}

/* PY Descourt 08/09/2009 */
void GateSourceVoxelRangeTranslator::UpdateActivity(G4double activmin , G4double activmax , G4double Updated_Activity)
{

  for (G4int iRange = 0; iRange< (G4int)m_voxelActivityTranslation.size(); iRange++) {
    G4double range1 = (m_voxelActivityTranslation[iRange].first).first;
    G4double range2 = (m_voxelActivityTranslation[iRange].first).second;
       G4cout << "activity range " << activmin<<"  "<< activmax <<"    iRange range1 range2 " << iRange << " " << range1 << " " << range2 << G4endl;
    if ( (  fabs(range1 - activmin) < 1e-8   ) &&(  fabs(range2 - activmax) < 1e-8   )   ) {
      m_voxelActivityTranslation[iRange].second = Updated_Activity;
      break;
    }
  }


}

void GateSourceVoxelRangeTranslator::AddTranslationRange( G4double rmin , G4double rmax ) 
{
    std::pair<G4double,G4double> minmax(rmin, rmax);
    GateVoxelActivityTranslationRange range(minmax, 0. );
    // Add check on possible overlaps with previously defined image value ranges
    // before adding this range to the range table
    
    GateVoxelActivityTranslationRangeVector::iterator it = find(m_voxelActivityTranslation.begin(),m_voxelActivityTranslation.end(),range);
    
    
    if ( it != m_voxelActivityTranslation.end() ) m_voxelActivityTranslation.push_back(range);
  }


/* PY Descourt 08/09/2009 */
