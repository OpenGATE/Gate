/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateArrayRepeater.hh"
#include "GateArrayRepeaterMessenger.hh"

#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4UnitsTable.hh"

#include "GateTools.hh"


//--------------------------------------------------------------------------------------------
GateArrayRepeater::GateArrayRepeater(GateVVolume* itsObjectInserter,
                                     const G4String& itsName,
                                     G4int itsRepeatNumberX,
                                     G4int itsRepeatNumberY,
                                     G4int itsRepeatNumberZ,
                                     const G4ThreeVector& itsRepeatVector,
                                     G4bool itsFlagAutoCenter)
  : GateVGlobalPlacement(itsObjectInserter,itsName),
    m_repeatVector(itsRepeatVector),
    m_flagAutoCenter(itsFlagAutoCenter),
    m_Messenger(0)
{
  m_repeatNumber[0] = itsRepeatNumberX;
  m_repeatNumber[1] = itsRepeatNumberY;
  m_repeatNumber[2] = itsRepeatNumberZ;

  m_Messenger = new GateArrayRepeaterMessenger(this);
}
//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
GateArrayRepeater::~GateArrayRepeater()
{  
  delete m_Messenger;
}
//--------------------------------------------------------------------------------------------
/*
  void GateArrayRepeater::PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
  const G4ThreeVector& currentPosition,
  G4double aTime)
*/
//--------------------------------------------------------------------------------------------
void GateArrayRepeater::PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
					 const G4ThreeVector& currentPosition,
					 G4double )
{
  
  G4double dx = m_repeatVector[0],
    dy = m_repeatVector[1],
    dz = m_repeatVector[2];

  G4int    nx = m_repeatNumber[0],
    ny = m_repeatNumber[1],
    nz = m_repeatNumber[2];

  G4ThreeVector firstCopyOffset(0.,0.,0.);

  if (GetAutoCenterFlag())
    firstCopyOffset = .5 * G4ThreeVector( ( 1 - nx ) * dx, ( 1 - ny ) * dy, ( 1 - nz ) * dz);  
      
  G4double x_0 = currentPosition[0] + firstCopyOffset[0],
    y_0 = currentPosition[1] + firstCopyOffset[1],
    z_0 = currentPosition[2] + firstCopyOffset[2];
   
  for ( G4int k=0 ; k < nz ; k++)
    for ( G4int j=0 ; j < ny ; j++)      
      for ( G4int i=0 ; i < nx ; i++) {
          	   
	G4double x = x_0 + i * dx,
	  y = y_0 + j * dy,
	  z = z_0 + k * dz;
	  
	G4ThreeVector newPosition = G4ThreeVector(x,y,z);
	   
	PushBackPlacement(currentRotationMatrix,newPosition);
      }
}
//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
void GateArrayRepeater::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Repetition type:         " << "cubicArray"   << "\n";
  G4cout << GateTools::Indent(indent) << "Nb of copies along X   : " << m_repeatNumber[0]   << "\n";
  G4cout << GateTools::Indent(indent) << "Nb of copies along Y   : " << m_repeatNumber[1]   << "\n";
  G4cout << GateTools::Indent(indent) << "Nb of copies along Z   : " << m_repeatNumber[2]   << "\n";
  G4cout << GateTools::Indent(indent) << "Repetition step along X: " << G4BestUnit(m_repeatVector[0],"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Repetition step along Y: " << G4BestUnit(m_repeatVector[1],"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Repetition step along Z: " << G4BestUnit(m_repeatVector[2],"Length") << "\n";
  G4cout << GateTools::Indent(indent) << "Centering:               " << ( GetAutoCenterFlag() ? "Yes" : "No" ) << "\n";
}
//--------------------------------------------------------------------------------------------
