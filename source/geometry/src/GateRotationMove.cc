/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateRotationMove.hh"
#include "GateRotationMoveMessenger.hh"

#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4UnitsTable.hh"

#include "GateTools.hh"

//-------------------------------------------------------------------------------------------------
GateRotationMove::GateRotationMove(GateVVolume* itsObjectInserter,
                                   const G4String& itsName,
                                   const G4ThreeVector& itsRotationAxis,
                                   double itsVelocity)
  : GateVGlobalPlacement(itsObjectInserter,itsName),
    m_rotationAxis(itsRotationAxis),
    m_velocity(itsVelocity),
    m_currentAngle(0),
    m_Messenger(0)
{
  m_Messenger = new GateRotationMoveMessenger(this);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
GateRotationMove::~GateRotationMove()
{  
  delete m_Messenger;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateRotationMove::PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
                                        const G4ThreeVector& currentPosition,
                                        G4double aTime)
{
  G4RotationMatrix newRotationMatrix = currentRotationMatrix;

  m_currentAngle = m_velocity * aTime;
  newRotationMatrix.rotate(-m_currentAngle, m_rotationAxis);

  PushBackPlacement(GatePlacement(newRotationMatrix,currentPosition));
    
  // return GatePlacement(newRotationMatrix,currentPosition);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateRotationMove::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Move type:          " << "rotation"   << "\n";
  G4cout << GateTools::Indent(indent) << "Rotation axis:          " << m_rotationAxis   << "\n";
  G4cout << GateTools::Indent(indent) << "Rotation Velocity:      " << m_velocity / (degree/s)    << " deg/s\n";
  G4cout << GateTools::Indent(indent) << "Current rotation angle: " << m_currentAngle / degree << " deg\n";
}
//-------------------------------------------------------------------------------------------------
