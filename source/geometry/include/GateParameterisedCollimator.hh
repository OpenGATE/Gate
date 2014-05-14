/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateParameterisedCollimator_h
#define GateParameterisedCollimator_h 1

#include "globals.hh"

#include "GateBox.hh"


class GateParameterisedHole;
class GateParameterisedCollimatorMessenger;


class GateParameterisedCollimator : public GateBox
{
  public:
  // Constructor1
     GateParameterisedCollimator(const G4String& itsName,G4bool acceptsChildren=true, 
		 			 G4int depth=0);
  // Constructor2
     GateParameterisedCollimator(const G4String& itsName,const G4String& itsMaterialName,
     G4double itsFocalDistanceX,G4double itsFocalDistanceY,G4double itsSeptalThickness,
     G4double itsInnerRadius,G4double itsHeight,G4double itsDimensionX,G4double itsDimensionY);

     virtual ~GateParameterisedCollimator();

     FCT_FOR_AUTO_CREATOR_VOLUME(GateParameterisedCollimator)
     
     GateBox* GetCollimatorCreator() const
      { return (GateBox*) GetCreator(); }

     void PreComputeConstants();

     void ResizeCollimator();

     inline G4double GetCollimatorFocalDistanceX() const
	{ return m_FocalDistanceX; }
     inline G4double GetCollimatorFocalDistanceY() const
	{ return m_FocalDistanceY; }
     inline G4double GetCollimatorSeptalThickness() const
	{ return m_SeptalThickness; }
     inline G4double GetCollimatorInnerRadius() const
	{ return m_InnerRadius; }
     inline G4double GetCollimatorHeight() const
	{ return m_Height; }
     inline G4double GetCollimatorDimensionX() const
	{ return m_DimensionX; }
     inline G4double GetCollimatorDimensionY() const
	{ return m_DimensionY; }
     inline const G4String& GetCollimatorMaterial() const
	{ return mMaterialName; }

     inline void SetCollimatorFocalDistanceX(G4double val)
	{ m_FocalDistanceX = val; ResizeCollimator();}
     inline void SetCollimatorFocalDistanceY(G4double val)
	{ m_FocalDistanceY = val; ResizeCollimator();}
     inline void SetCollimatorSeptalThickness(G4double val)
	{ m_SeptalThickness = val; ResizeCollimator();}
     inline void SetCollimatorInnerRadius(G4double val)
      	{ m_InnerRadius = val; ResizeCollimator(); }
     inline void SetCollimatorHeight(G4double val)
	{ m_Height = val; ResizeCollimator(); }
     inline void SetCollimatorDimensionX(G4double val)
	{ m_DimensionX = val; ResizeCollimator(); }
     inline void SetCollimatorDimensionY(G4double val)
	{ m_DimensionY = val; ResizeCollimator();}
     inline void SetCollimatorMaterial(const G4String& val)
	{ mMaterialName = val; ResizeCollimator();}

  protected:
     G4double m_FocalDistanceX,m_FocalDistanceY,m_SeptalThickness,m_InnerRadius,m_Height,m_DimensionX,m_DimensionY;
     G4String mMaterialName;

     GateParameterisedHole*                m_holeInserter;
     GateParameterisedCollimatorMessenger* m_messenger;
};

MAKE_AUTO_CREATOR_VOLUME(collimator,GateParameterisedCollimator)

#endif
