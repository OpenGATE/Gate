/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateEccentRotMove.hh"
#include "GateEccentRotMoveMessenger.hh"

#include "GateVGlobalPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4UnitsTable.hh"

#include "GateTools.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

//------------------------------------------------------------------------------------------------
GateEccentRotMove::GateEccentRotMove(GateVVolume* itsObjectInserter,
      	      	      	      	      	 const G4String& itsName,
     	      	      	      	      	 const G4ThreeVector& itsShift,
				      	 double itsVelocity)
  : GateVGlobalPlacement(itsObjectInserter,itsName),
    m_shift(itsShift),
    m_velocity(itsVelocity),
    m_currentAngle(0),
    m_Messenger(0)
{
  m_Messenger = new GateEccentRotMoveMessenger(this);
}
//------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------
GateEccentRotMove::~GateEccentRotMove()
{  
  delete m_Messenger;
}
//------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------
void GateEccentRotMove::PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
	        	        	      	      	      	      	    const G4ThreeVector& currentPosition,
			      	      	      	      	      	      	    G4double aTime)
{
    G4RotationMatrix newRotationMatrix = currentRotationMatrix;

    // First shift to eccentric position in the Z=0 plane
    G4Point3D newPosition = currentPosition+ m_shift;

    // Then Make an orbiting Move around OZ axis with autorotation
    m_currentAngle = m_velocity * aTime;
    //    newPosition = G4RotateZ3D(m_currentAngle,m_point1,m_point2) * newPosition;
    newPosition = G4RotateZ3D(m_currentAngle) * newPosition;
    
    //    G4Point3D m_point1(0.,0.,0.), m_point2(0.,0.,1.);
    // G4ThreeVector axis = m_point2 - m_point1;

    G4ThreeVector axis(0.,0.,1.);
    newRotationMatrix.rotate(-m_currentAngle, axis);    // Autorotate the object around OZ

    PushBackPlacement(GatePlacement(newRotationMatrix,newPosition));  
//    return GatePlacement(newRotationMatrix,newPosition);
}
//------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------
void GateEccentRotMove::DescribeMyself(size_t indent)
{
    G4cout << GateTools::Indent(indent) << "Move type:          " << "XYZ translation, then orbiting around OZ"   << "\n";
    G4cout << GateTools::Indent(indent) << "XYZ Shift value to eccentric positions :          " << m_shift << "\n";
    G4cout << GateTools::Indent(indent) << "EccentRot Velocity:      " << m_velocity / (degree/s)    << " deg/s\n";
    G4cout << GateTools::Indent(indent) << "Current orbiting angle: " << m_currentAngle / degree << " deg\n";
}
//------------------------------------------------------------------------------------------------




