/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateOscTranslationMove.hh"
#include "GateOscTranslationMoveMessenger.hh"

#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4UnitsTable.hh"

#include "GateTools.hh"

//-------------------------------------------------------------------------------------------------
GateOscTranslationMove::GateOscTranslationMove(GateVVolume* itsObjectInserter,
      	      	      	      	      	       const G4String& itsName,
				      	       const G4ThreeVector& itsAmplitude,
				      	       G4double itsFrequency,
				      	       G4double itsPhase )
  : GateVGlobalPlacement(itsObjectInserter,itsName),
    m_amplitude(itsAmplitude),
    m_frequency(itsFrequency),
    m_phase(itsPhase),
    m_currentTranslation(0.),
    m_Messenger(0)
{
  m_Messenger = new GateOscTranslationMoveMessenger(this);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
GateOscTranslationMove::~GateOscTranslationMove()
{  
  delete m_Messenger;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateOscTranslationMove::PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
	        	        	      const G4ThreeVector& currentPosition,
			      	      	      G4double aTime)
{
    m_currentTranslation =  m_amplitude * sin ( 2. * M_PI * m_frequency * aTime + m_phase);
    G4ThreeVector newPosition = currentPosition + m_currentTranslation;

    PushBackPlacement(GatePlacement(currentRotationMatrix,newPosition)); 
//    return GatePlacement(currentRotationMatrix,newPosition);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateOscTranslationMove::DescribeMyself(size_t indent)
{
    G4cout << GateTools::Indent(indent) << "Move type:            " << "oscillating translation"   << G4endl;
    G4cout << GateTools::Indent(indent) << "Maximum amplitude:    " << G4BestUnit(m_amplitude,"Length") << G4endl;
    G4cout << GateTools::Indent(indent) << "Frequency:            " << G4BestUnit(m_frequency,"Frequency") << G4endl;
    G4cout << GateTools::Indent(indent) << "Period:               " << G4BestUnit(GetPeriod(),"Time") << G4endl;
    G4cout << GateTools::Indent(indent) << "Phase:                " << m_phase / degree << " deg" << G4endl;
    G4cout << GateTools::Indent(indent) << "Current translation:  " << G4BestUnit(m_currentTranslation,"Length") << G4endl;
}
//-------------------------------------------------------------------------------------------------
