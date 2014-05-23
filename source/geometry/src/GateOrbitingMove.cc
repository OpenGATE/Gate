/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "Randomize.hh"

#include "GateOrbitingMove.hh"
#include "GateOrbitingMoveMessenger.hh"

#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4UnitsTable.hh"

#include "GateTools.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

//-------------------------------------------------------------------------------------------------
GateOrbitingMove::GateOrbitingMove(GateVVolume* itsObjectInserter,
      	      	      	      	   const G4String& itsName,
     	      	      	      	   const G4Point3D& itsPoint1,
     	      	      	      	   const G4Point3D& itsPoint2,
				   double itsVelocity,
      	      	              	   G4bool itsFlagAutoRotation)
  : GateVGlobalPlacement(itsObjectInserter,itsName),
    m_point1(itsPoint1),
    m_point2(itsPoint2),
    m_velocity(itsVelocity),
    m_flagAutoRotation(itsFlagAutoRotation),
    m_currentAngle(0),
    m_Messenger(0)
{
  
  m_Messenger = new GateOrbitingMoveMessenger(this);
  
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
GateOrbitingMove::~GateOrbitingMove()
{  
  delete m_Messenger;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateOrbitingMove::PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
	        	        	const G4ThreeVector& currentPosition,
			      	      	G4double aTime)
{
    
    G4RotationMatrix newRotationMatrix = currentRotationMatrix;
    G4Point3D newPosition = currentPosition;
        	
    m_currentAngle = m_velocity * aTime;
    newPosition = G4Rotate3D(m_currentAngle, m_point1, m_point2) * newPosition;
        
    if (m_flagAutoRotation)
    {
      G4ThreeVector axis = m_point2 - m_point1;
      newRotationMatrix.rotate(-m_currentAngle, axis);
    }

    PushBackPlacement(GatePlacement(newRotationMatrix,newPosition));
    
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateOrbitingMove::DescribeMyself(size_t indent)
{
    G4cout << GateTools::Indent(indent) << "Move type:          " << "orbiting"   << "\n";
    G4cout << GateTools::Indent(indent) << "Orbiting axis:          " << m_point1 << " <--> " << m_point2   << "\n";
    G4cout << GateTools::Indent(indent) << "Orbiting Velocity:      " << m_velocity / (degree/s)    << " deg/s\n";
    G4cout << GateTools::Indent(indent) << "Current orbiting angle: " << m_currentAngle / degree << " deg\n";
    G4cout << GateTools::Indent(indent) << "Auto-rotation:          " << GetAutoRotation()  << "\n";
}
//-------------------------------------------------------------------------------------------------
