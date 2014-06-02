/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateLinearRepeater.hh"
#include "GateLinearRepeaterMessenger.hh"

#include "GateVVolume.hh"

#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4UnitsTable.hh"

#include "GateTools.hh"

//--------------------------------------------------------------------------------------------------------------
GateLinearRepeater::GateLinearRepeater(GateVVolume* itsObjectInserter,
      	      	      	      	       const G4String& itsName,
      	      	      	      	       G4int itsRepeatNumber,
      	      	      	      	       const G4ThreeVector& itsRepeatVector,
		              	       G4bool itsFlagAutoCenter)
  : GateVGlobalPlacement(itsObjectInserter,itsName),
    m_repeatVector(itsRepeatVector),
    m_repeatNumber(itsRepeatNumber),
    m_flagAutoCenter(itsFlagAutoCenter),
    m_Messenger(0)
{
  m_Messenger = new GateLinearRepeaterMessenger(this);
}
//--------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------
GateLinearRepeater::~GateLinearRepeater()
{  
  delete m_Messenger;
}
//--------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------
void GateLinearRepeater::PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
	        	      	      	  const G4ThreeVector& currentPosition,
			      	      	  G4double )
{
 
  G4ThreeVector firstCopyOffset(0.,0.,0.);
  
  if (GetAutoCenterFlag())
      firstCopyOffset = m_repeatVector * .5 * ( 1 - m_repeatNumber );
  
  for ( G4int i=0 ; i < m_repeatNumber ; i++) {
   
    
    G4ThreeVector newPosition = currentPosition + firstCopyOffset + m_repeatVector * i ;
   
    PushBackPlacement(currentRotationMatrix,newPosition);
  }

}
//--------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------
void GateLinearRepeater::DescribeMyself(size_t indent)
{
    G4cout << GateTools::Indent(indent) << "Repetiton type:    " << "linear"   << "\n";
    G4cout << GateTools::Indent(indent) << "Repetition vector: " << G4BestUnit(m_repeatVector,"Length") << "\n";
    G4cout << GateTools::Indent(indent) << "Nb of copies:      " << m_repeatNumber   << "\n";
    G4cout << GateTools::Indent(indent) << "Centering:         " << ( GetAutoCenterFlag() ? "Yes" : "No" ) << "\n";
}
//--------------------------------------------------------------------------------------------------------------
