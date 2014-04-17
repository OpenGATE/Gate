/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateAngularRepeater.hh"
#include "GateAngularRepeaterMessenger.hh"
#include "GateTools.hh"

#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4Transform3D.hh"
#include "G4UnitsTable.hh"

//-----------------------------------------------------------------------------------------------
GateAngularRepeater::GateAngularRepeater(GateVVolume* itsObjectInserter,
					 const G4String& itsName,
					 G4int itsRepeatNumber,
					 const G4Point3D& itsPoint1,
					 const G4Point3D& itsPoint2,
					 G4bool   itsFlagAutoRotation,
					 G4double itsFirstAngle,
					 G4double itsAngularSpan,
					 G4int    itsModuloNumber,
					 G4double itsZShift1,
					 G4double itsZShift2,
					 G4double itsZShift3,
					 G4double itsZShift4,
					 G4double itsZShift5,
					 G4double itsZShift6,
					 G4double itsZShift7,
					 G4double itsZShift8)
: GateVGlobalPlacement(itsObjectInserter,itsName),
  m_repeatNumber(itsRepeatNumber),
  m_point1(itsPoint1),
  m_point2(itsPoint2),
  m_flagAutoRotation(itsFlagAutoRotation),
  m_firstAngle(itsFirstAngle),
  m_angularSpan(itsAngularSpan),
  m_moduloNumber(itsModuloNumber),
  m_zShift1(itsZShift1),
  m_zShift2(itsZShift2),
  m_zShift3(itsZShift3),
  m_zShift4(itsZShift4),
  m_zShift5(itsZShift5),
  m_zShift6(itsZShift6),
  m_zShift7(itsZShift7),
  m_zShift8(itsZShift8),
  m_Messenger(0)
{
  m_Messenger = new GateAngularRepeaterMessenger(this);
}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
GateAngularRepeater::~GateAngularRepeater()
{
  delete m_Messenger;
}
//-----------------------------------------------------------------------------------------------



//-----------------------------------------------------------------------------------------------
void GateAngularRepeater::PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
					   const G4ThreeVector& currentPosition,
					   G4double )
{
  G4double dphi ;
  if ( m_angularSpan < 360.0005*deg && m_angularSpan > 359.9995*deg)
    dphi = GetAngularPitch_1();
  else
    dphi = GetAngularPitch_2();

  G4ThreeVector axis = m_point2 - m_point1;
  //G4int modulo = m_moduloNumber;
  G4ThreeVector ShiftVector;

  G4double Zshift_vector[] = {
    m_zShift1,
    m_zShift2,
    m_zShift3,
    m_zShift4,
    m_zShift5,
    m_zShift6,
    m_zShift7,
    m_zShift8
  };

  if ( m_moduloNumber > ModuloMax ){
    G4cout << "\n GateAngularRepeater says: *** WARNING *** Modulo number > ModuloMax change value to 1 !! \n";
    m_moduloNumber = 1 ;}

  //      for ( i= m_moduloNumber+1; i <= ModuloMax ; i++) {
  //      if(Zshift_vector[i] > 0.)
  //      G4cout << "\n GateAngularRepeater says: macro line " << "setZShift" << i <<" "<<Zshift_vector[i]<<" ...\n"
  //      " with i greater than "<<m_moduloNumber<<" is not considered !!" <<
  //      "\n for bigger values, please change #define ModuloMax [max]\n in file GateAngularRepeater.h\n";}

  for ( G4int i=0 ; i < GetRepeatNumber() ; i++) {

    G4RotationMatrix newRotationMatrix = currentRotationMatrix;
    G4Point3D m_shift = G4ThreeVector(0., 0., Zshift_vector[i%m_moduloNumber]); // Pick up right value


    G4Point3D newPosition = currentPosition + m_shift;


    G4double angle = m_firstAngle + dphi*i;


    newPosition = HepGeom::Rotate3D(angle,m_point1,m_point2) * newPosition;

    if (m_flagAutoRotation)
      newRotationMatrix.rotate(-angle, axis);

    PushBackPlacement(newRotationMatrix,newPosition);
  }

}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
void GateAngularRepeater::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Repetition type:      " << "Modulo M"   << "\n";
  G4cout << GateTools::Indent(indent) << "Nb of copies:         " << m_repeatNumber   << "\n";
  G4cout << GateTools::Indent(indent) << "Repetition axis:      " << m_point1 /cm << " <--> " << m_point2  /cm  << "cm\n";
  G4cout << GateTools::Indent(indent) << "Auto-rotation:        " << GetAutoRotation()  << "\n";
  G4cout << GateTools::Indent(indent) << "First rotation angle: " << m_firstAngle / degree  << " deg\n";
  G4cout << GateTools::Indent(indent) << "Angular span:         " << m_angularSpan / degree  << " deg\n";
  G4cout << GateTools::Indent(indent) << "Angular pitch:        " << GetAngularPitch_1() / degree  << " deg\n";
  G4cout << GateTools::Indent(indent) << "Modulo nomber:        " << m_moduloNumber << "\n";
  G4cout << GateTools::Indent(indent) << "Z shift 1 modulo N:   " << m_zShift1 /cm << "cm\n";
  G4cout << GateTools::Indent(indent) << "Z shift 2 modulo N:   " << m_zShift2 /cm << "cm\n";
  G4cout << GateTools::Indent(indent) << "Z shift 3 modulo N:   " << m_zShift3 /cm << "cm\n";
  G4cout << GateTools::Indent(indent) << "Z shift 4 modulo N:   " << m_zShift4 /cm << "cm\n";
  G4cout << GateTools::Indent(indent) << "Z shift 5 modulo N:   " << m_zShift5 /cm << "cm\n";
  G4cout << GateTools::Indent(indent) << "Z shift 6 modulo N:   " << m_zShift6 /cm << "cm\n";
  G4cout << GateTools::Indent(indent) << "Z shift 7 modulo N:   " << m_zShift7 /cm << "cm\n";
  G4cout << GateTools::Indent(indent) << "Z shift 8 modulo N:   " << m_zShift8 /cm << "cm\n";
}
//-----------------------------------------------------------------------------------------------
