/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateTranslationMove.hh"
#include "GateTranslationMoveMessenger.hh"

#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4UnitsTable.hh"

#include "GateTools.hh"

//-------------------------------------------------------------------------------------------------
GateTranslationMove::GateTranslationMove(GateVVolume* itsObjectInserter,
      	      	      	      	      	      	 const G4String& itsName,
				      	      	 const G4ThreeVector& itsVelocity )
  : GateVGlobalPlacement(itsObjectInserter,itsName),
    m_velocity(itsVelocity),
    m_currentTranslation(0.),
    m_Messenger(0)
{
  m_Messenger = new GateTranslationMoveMessenger(this);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
GateTranslationMove::~GateTranslationMove()
{  
  delete m_Messenger;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateTranslationMove::PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
	        	        	      	      	      	      	       const G4ThreeVector& currentPosition,
			      	      	      	      	      	      	       G4double aTime)
{
    m_currentTranslation =  m_velocity * aTime ;
    G4ThreeVector newPosition = currentPosition + m_currentTranslation;

    PushBackPlacement(GatePlacement(currentRotationMatrix,newPosition));  
//    return GatePlacement(currentRotationMatrix,newPosition);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateTranslationMove::DescribeMyself(size_t indent)
{
    G4cout << GateTools::Indent(indent) << "Move type:        " << "translation"   << "\n";
    G4cout << GateTools::Indent(indent) << "Translation Velocity: " << G4BestUnit(m_velocity,"Speed") << "\n";
    G4cout << GateTools::Indent(indent) << "Current translation:  " << G4BestUnit(m_currentTranslation,"Length") << "\n";
}
//-------------------------------------------------------------------------------------------------
