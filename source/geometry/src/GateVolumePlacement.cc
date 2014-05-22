/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVolumePlacement.hh"
#include "GateVolumePlacementMessenger.hh"
#include "G4Transform3D.hh"
#include "G4UnitsTable.hh"
#include "GateVVolume.hh"
#include "GateTools.hh"

//-----------------------------------------------------------------------------------------------
GateVolumePlacement::GateVolumePlacement(GateVVolume* itsObjectInserter,
      	      	      	      	      	     const G4String& itsName,
				      	     const G4ThreeVector& itsTranslation,
				      	     const G4ThreeVector itsRotationAxis,
				      	     G4double itsRotationAngle)
  : GateVGlobalPlacement(itsObjectInserter,itsName),
    m_translation(itsTranslation),
    m_rotationAxis(itsRotationAxis),
    m_rotationAngle(itsRotationAngle),
    m_Messenger(0)
{
    m_Messenger = new GateVolumePlacementMessenger(this);
}
//-----------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------------
GateVolumePlacement::~GateVolumePlacement()
{  
  if (m_Messenger) delete m_Messenger;
}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
void GateVolumePlacement::PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
                                           const G4ThreeVector& currentPosition,
                                           G4double )
{
    G4RotationMatrix newRotationMatrix = currentRotationMatrix;
    newRotationMatrix.rotate(-m_rotationAngle, m_rotationAxis);    
    G4ThreeVector newPosition = currentPosition + m_translation;    
    PushBackPlacement(GatePlacement(newRotationMatrix,newPosition));    
    //    return GatePlacement(newRotationMatrix,newPosition);
}
//-----------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------
void GateVolumePlacement::DescribeMyself(size_t indent)
{
    G4cout << GateTools::Indent(indent) << "Move type:     " << "placement"   << "\n";
    G4cout << GateTools::Indent(indent) << "Translation:       " << G4BestUnit(m_translation,"Length") << "\n";
    G4cout << GateTools::Indent(indent) << "Rotation axis:     " << m_rotationAxis   << "\n";
    G4cout << GateTools::Indent(indent) << "Rotation angle:    " << m_rotationAngle / degree << " deg\n";
}
//-----------------------------------------------------------------------------------------------
