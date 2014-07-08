/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateQuadrantRepeater.hh"
#include "GateQuadrantRepeaterMessenger.hh"

#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4UnitsTable.hh"

#include "GateTools.hh"

GateQuadrantRepeater::GateQuadrantRepeater(GateVVolume* itsObjectInserter,
      	      	      	      	       const G4String& itsName,
      	      	      	      	       G4int itsLineNumber,
      	      	      	      	       G4double itsOrientation,
				       G4double itsCopySpacing)
  : GateVGlobalPlacement(itsObjectInserter,itsName),
    m_lineNumber(itsLineNumber),
    m_orientation(itsOrientation),
    m_copySpacing(itsCopySpacing),
    m_maxRange(DBL_MAX),
    m_Messenger(0)
{
  ComputeRepetitionVectors();

  m_Messenger = new GateQuadrantRepeaterMessenger(this);
}




GateQuadrantRepeater::~GateQuadrantRepeater()
{  
  delete m_Messenger;
}

/*
void GateQuadrantRepeater::PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
	        	      	      	  const G4ThreeVector& currentPosition,
			      	      	  G4double aTime
*/
void GateQuadrantRepeater::PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
	        	      	      	  const G4ThreeVector& currentPosition,
			      	      	  G4double )
{
  for ( G4int i=0 ; i < m_lineNumber ; i++) {

    G4int orthogonalRepeatNumber = i+1;

    G4ThreeVector startPosition = currentPosition + m_lineSpacingVector * i;
    
    for ( G4int j=0 ; j<orthogonalRepeatNumber ; j++) {

      G4ThreeVector newPosition = startPosition + j * m_orthogonalRepeatVector;

      G4double dist = sqrt( newPosition.diff2(currentPosition) );
      if ( dist <= m_maxRange)
      	PushBackPlacement(currentRotationMatrix,newPosition);
    }
  }

}




void GateQuadrantRepeater::DescribeMyself(size_t indent)
{
    G4cout << GateTools::Indent(indent) << "Repeater type:          " << "quadrant"   << "\n";
    G4cout << GateTools::Indent(indent) << "Nb of lines:            " << m_lineNumber   << "\n";
    G4cout << GateTools::Indent(indent) << "Orientation:            " << m_orientation/degree << " deg\n";
    G4cout << GateTools::Indent(indent) << "Object spacing:         " << G4BestUnit(m_copySpacing,"Length") << "\n";
    G4cout << GateTools::Indent(indent) << "Maximum range:          " << G4BestUnit(m_maxRange,"Length") << "\n";
    G4cout << GateTools::Indent(indent) << "Line-spacing vector:    " << G4BestUnit(m_lineSpacingVector,"Length") << "\n";
    G4cout << GateTools::Indent(indent) << "Repetition vector:      " << G4BestUnit(m_orthogonalRepeatVector,"Length") << "\n";
}

