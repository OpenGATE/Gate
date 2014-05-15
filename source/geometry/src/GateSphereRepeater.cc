/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSphereRepeater.hh"
#include "GateSphereRepeaterMessenger.hh"

#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4UnitsTable.hh"

#include "GateTools.hh"

GateSphereRepeater::GateSphereRepeater(GateVVolume* itsObjectInserter,
      	      	      	      	       const G4String& itsName,
      	      	      	      	       G4int itsRepeatNumberWithPhi,
      	      	      	      	       G4int itsRepeatNumberWithTheta,
				       G4double itsThetaAngle,
				       G4double itsPhiAngle,
				       G4double itsRadius,
		              	       G4bool itsFlagAutoCenter,
				       G4bool   itsFlagAutoRotation )
  : GateVGlobalPlacement(itsObjectInserter,itsName),
    m_repeatNumberWithPhi(itsRepeatNumberWithPhi),
    m_repeatNumberWithTheta(itsRepeatNumberWithTheta),
    m_thetaAngle(itsThetaAngle),
    m_phiAngle(itsPhiAngle),
    m_radius(itsRadius),
    m_flagAutoCenter(itsFlagAutoCenter),
    m_flagAutoRotation(itsFlagAutoRotation),
    m_Messenger(0)
{
  m_Messenger = new GateSphereRepeaterMessenger(this);
}




GateSphereRepeater::~GateSphereRepeater()
{  
  delete m_Messenger;
}




void GateSphereRepeater::PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
	        	      	      	      const G4ThreeVector& currentPosition,
			      	      	      G4double )
{
  G4double radius = m_radius;
  G4double theta = m_thetaAngle;
  G4double phi = m_phiAngle;

  G4int    nPhi = m_repeatNumberWithPhi,
	   nTheta = m_repeatNumberWithTheta;

  G4ThreeVector firstCopyOffset(0.,0.,0.);
      
  G4double x_0 = currentPosition[0] + firstCopyOffset[0],
      	   y_0 = currentPosition[1] + firstCopyOffset[1],
	   z_0 = currentPosition[2] + firstCopyOffset[2];

  G4double phi_0 = -phi;
  G4double theta_j;

  for ( G4int k=0 ; k < nPhi ; k++)
    {
      G4double phi_k = phi_0 + k * phi;

    for ( G4int j=0 ; j < nTheta ; j++) 
      {
	theta_j = j * theta;
	G4RotationMatrix newRotationMatrix =  currentRotationMatrix; 

	G4double x = x_0 + radius*sin(theta_j)*cos(phi_k),
	         y = y_0 + radius*cos(theta_j)*cos(phi_k),
	         z = z_0 + radius*sin(phi_k);
	  
      	G4ThreeVector newPosition = G4ThreeVector(x,y,z);
	if (m_flagAutoRotation) {
	  newRotationMatrix.rotateZ(theta_j);
	  newRotationMatrix.rotateX(-phi_k);
	  }

      	PushBackPlacement(newRotationMatrix,newPosition);
      }
    }

}




void GateSphereRepeater::DescribeMyself(size_t indent)
{
    G4cout << GateTools::Indent(indent) << "Repetition type:             " << "sphere"   << "\n";
    G4cout << GateTools::Indent(indent) << "Nb of copies along the ring: " << m_repeatNumberWithTheta   << "\n";
    G4cout << GateTools::Indent(indent) << "Nb of copies along the normal direction   : " << m_repeatNumberWithPhi  << "\n";
    G4cout << GateTools::Indent(indent) << "Theta angle:                 " << GetThetaAngle() / degree  << " deg\n";
    G4cout << GateTools::Indent(indent) << "Phi angle:                   " << GetPhiAngle() / degree  << " deg\n";
    G4cout << GateTools::Indent(indent) << "Sphere Radius :          " << G4BestUnit(GetRadius(),"Length") << "\n";
    G4cout << GateTools::Indent(indent) << "Centering:               " << ( GetAutoCenterFlag() ? "Yes" : "No" ) << "\n";
    G4cout << GateTools::Indent(indent) << "Auto-rotation:        " << GetAutoRotation()  << "\n";
}


