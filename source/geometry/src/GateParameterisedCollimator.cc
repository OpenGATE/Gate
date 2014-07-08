/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateParameterisedCollimator.hh"
#include "GateParameterisedCollimatorMessenger.hh"

#include "GateBox.hh"
#include "GateParameterisedHole.hh"
#include "GateObjectChildList.hh"
#include "GateMaterialDatabase.hh"

#include "G4UnitsTable.hh"
#include "G4VisAttributes.hh"

//-------------------------------------------------------------------------------------------------------------------
GateParameterisedCollimator::GateParameterisedCollimator(const G4String& itsName,
							 G4bool acceptsChildren, 
		 			 		 G4int depth)
: GateBox(itsName,"Vacuum",41.,22.,4.,acceptsChildren,depth),
     m_FocalDistanceX(39.7*cm),m_FocalDistanceY(0.0*cm),
     m_SeptalThickness(0.1* cm),m_InnerRadius(0.05*cm),m_Height(4.*cm),
     m_DimensionX(41.0*cm),m_DimensionY(22.0*cm),
     mMaterialName("Vacuum")
{ 
  G4cout << " Constructor GateParameterisedCollimator - begin " << itsName << G4endl;
  
  G4cout << " m_InnerRadius = " << m_InnerRadius << G4endl;
  
  m_holeInserter = new GateParameterisedHole("hole","Air",m_FocalDistanceX,m_FocalDistanceY,m_SeptalThickness,
  		                                          m_InnerRadius,m_Height,m_DimensionX,m_DimensionY);
  
  GetCreator()->GetTheChildList()->AddChild(m_holeInserter);

  m_messenger = new GateParameterisedCollimatorMessenger(this);
  
  G4cout << " Constructor GateParameterisedCollimator - end " << G4endl;
}
//-------------------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------------------
GateParameterisedCollimator::GateParameterisedCollimator(const G4String& itsName,const G4String& itsMaterialName,
     G4double itsFocalDistanceX,G4double itsFocalDistanceY,G4double itsSeptalThickness,G4double itsInnerRadius,
     G4double itsHeight,G4double itsDimensionX,G4double itsDimensionY)
     : GateBox(itsName,itsMaterialName,itsDimensionX,itsDimensionY,itsHeight,false,false),
     m_FocalDistanceX(itsFocalDistanceX),m_FocalDistanceY(itsFocalDistanceY),
     m_SeptalThickness(itsSeptalThickness),m_InnerRadius(itsInnerRadius),m_Height(itsHeight),
     m_DimensionX(itsDimensionX),m_DimensionY (itsDimensionY),
     mMaterialName(itsMaterialName)
{
  m_holeInserter = new GateParameterisedHole("hole","Air",m_FocalDistanceX,m_FocalDistanceY,m_SeptalThickness,
  		m_InnerRadius,m_Height,m_DimensionX,m_DimensionY);
  GetCreator()->GetTheChildList()->AddChild(m_holeInserter);

  m_messenger = new GateParameterisedCollimatorMessenger(this);
}
//-------------------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------------------
GateParameterisedCollimator::~GateParameterisedCollimator()
{
  delete m_messenger;
}
//-------------------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------------------
void GateParameterisedCollimator::ResizeCollimator()
{
  GetCollimatorCreator()->SetBoxXLength(m_DimensionX);
  GetCollimatorCreator()->SetBoxYLength(m_DimensionY);
  GetCollimatorCreator()->SetBoxZLength(m_Height);
  GetCollimatorCreator()->SetMaterialName(mMaterialName);

  m_holeInserter->ResizeHole(m_FocalDistanceX,m_FocalDistanceY,m_SeptalThickness,m_InnerRadius,
			     m_Height,m_DimensionX,m_DimensionY);
}
//-------------------------------------------------------------------------------------------------------------------
