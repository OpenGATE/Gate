/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

// GateParameterisedPinholeCollimator class for analytic parameterization 
// Contact: Olga Kochebina kochebina@gmail.com


#ifndef GateParameterisedPinholeCollimator_h
#define GateParameterisedPinholeCollimator_h 1

#include "globals.hh"

#include "GateTrpd.hh"

#include "G4Cons.hh"
#include "G4SubtractionSolid.hh"

class GateParameterisedPinholeCollimatorMessenger;


class GateParameterisedPinholeCollimator : public GateTrpd
{
  public:
  // Constructor1
     GateParameterisedPinholeCollimator(const G4String& itsName,G4bool acceptsChildren=true, 
		 			 G4int depth=0);
  // Constructor2
     GateParameterisedPinholeCollimator(const G4String& itsName,const G4String& itsMaterialName,
					const G4String& itsInputFile,
					G4double itsHeight,G4double itsRotRadius,
					G4double itsDimensionX1,G4double itsDimensionY1,
					G4double itsDimensionX2,G4double itsDimensionY2,
					G4bool itsTest, G4double itsPinholeDia);

     virtual ~GateParameterisedPinholeCollimator();

     FCT_FOR_AUTO_CREATOR_VOLUME(GateParameterisedPinholeCollimator)
     
    

     virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material*, G4bool);
  
  virtual void DestroyOwnSolidAndLogicalVolume();
  
  void PreComputeConstants();

  void ResizeCollimator();

 

  inline G4String GetCollimatorInputFile() const
  { return m_InputFile; }
  inline G4double GetCollimatorHeight() const
  { return m_Height; }
  inline G4double GetCollimatorRotRadius() const
  { return m_RotRadius; }
  inline G4double GetCollimatorDimensionX1() const
  { return m_DimensionX1; }
  inline G4double GetCollimatorDimensionY1() const
  { return m_DimensionY1; }
  inline G4double GetCollimatorDimensionX2() const
  { return m_DimensionX2; }
  inline G4double GetCollimatorDimensionY2() const
  { return m_DimensionY2; }


  inline void SetCollimatorInputFile(G4String val)
  { m_InputFile = val; ResizeCollimator(); }
  inline void SetCollimatorHeight(G4double val)
  { m_Height = val; ResizeCollimator(); }
  inline void SetCollimatorRotRadius(G4double val)
  { m_RotRadius = val; }
  inline void SetCollimatorDimensionX1(G4double val)
  { m_DimensionX1 = val; ResizeCollimator(); }
  inline void SetCollimatorDimensionY1(G4double val)
  { m_DimensionY1 = val; ResizeCollimator();}
  inline void SetCollimatorDimensionX2(G4double val)
  { m_DimensionX2 = val; ResizeCollimator(); }
  inline void SetCollimatorDimensionY2(G4double val)
  { m_DimensionY2 = val; ResizeCollimator();}

private:

  G4Trd*               m_colli_solid;     
  G4LogicalVolume*     m_colli_log; 

  G4Cons*  m_cone_up_solid;     
  G4Cons*  m_cone_down_solid; 

  G4SubtractionSolid*  m_sub_up_solid;     
  G4SubtractionSolid*  m_sub_down_solid; 

  G4LogicalVolume*     m_colli_pinholes_log; 	      	  

protected:
  G4double m_Height,m_RotRadius,m_DimensionX1,m_DimensionY1,m_DimensionX2,m_DimensionY2;
  G4String m_InputFile;

  GateParameterisedPinholeCollimatorMessenger* m_messenger;
};

MAKE_AUTO_CREATOR_VOLUME(pinhole_collimator,GateParameterisedPinholeCollimator)

#endif
