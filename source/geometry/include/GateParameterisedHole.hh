/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateParameterisedHole_h
#define GateParameterisedHole_h 1

#include "globals.hh"

#include "GateHoleParameterisation.hh"
#include "GateTrap.hh"


class GateHoleParameterisation;

class GateParameterisedHole : public GateTrap
{
  public:
     GateParameterisedHole(const G4String& itsName,const G4String& itsMaterialName,
      G4double itsFocalDistanceX,G4double itsFocalDistanceY,G4double itsSeptalThickness,
      G4double itsInnerRadius,G4double itsHeight,G4double itsDimensionX,G4double itsDimensionY);

     virtual ~GateParameterisedHole();

     GateTrap* GetHoleCreator() const
      { return (GateTrap*) GetCreator(); }

     GateHoleParameterisation* GetHoleParameterisation() const
      { return (GateHoleParameterisation*) GetParameterisation(); }

     void PreComputeConstants();

     void ResizeHole(G4double FocalDistanceX,G4double FocalDistanceY,G4double SeptalThickness,
      G4double InnerRadius,G4double Height,G4double DimensionX,G4double DimensionY);

     inline G4double GetHoleFocalDistanceX() const
      { return m_FocalDistanceX; }
     inline G4double GetHoleFocalDistanceY() const
      { return m_FocalDistanceY; }
     inline G4double GetHoleSeptalThickness() const
      { return m_SeptalThickness; }
     inline G4double GetHoleInnerRadius() const
	{ return m_InnerRadius; }
     inline G4double GetHoleHeight() const
	{ return m_Height; }
     inline G4double GetHoleDimensionX() const
	{ return m_DimensionX; }
     inline G4double GetHoleDimensionY() const
	{ return m_DimensionY; }

     inline void SetHoleFocalDistanceX(G4double val)
	{ m_FocalDistanceX = val; }
     inline void SetHoleFocalDistanceY(G4double val)
	{ m_FocalDistanceY = val; }
     inline void SetHoleSeptalThickness(G4double val)
	{ m_SeptalThickness = val; }
     inline void SetHoleInnerRadius(G4double val)
      	{ m_InnerRadius = val; }
     inline void SetHoleHeight(G4double val)
	{ m_Height = val; }
     inline void SetHoleDimensionX(G4double val)
	{ m_DimensionX = val; }
     inline void SetHoleDimensionY(G4double val)
	{ m_DimensionY = val; }



     //! Implementation of the pure virtual method declared by the base class GateVCreator
     //! If flagUpdateOnly is 0, it creates a new parameterised, using the parameterisation
     //! This parameterisation must have been created by a concrete class derived from GateVParameterised
     //! If flagUpdateOnly is set to 1, the default position is updated 
     virtual void ConstructOwnPhysicalVolume(G4bool flagUpdateOnly);
     
     //! Return the parameterisation
     virtual inline GatePVParameterisation* GetParameterisation() const
      	{ return m_parameterisation; }
     //! Set the parameterisation
     virtual inline void SetParameterisation(GatePVParameterisation* aParameterisation) 
      	{ m_parameterisation = aParameterisation; }
	
  protected:

     G4double m_FocalDistanceX,m_FocalDistanceY,m_SeptalThickness,m_InnerRadius,m_Height,m_DimensionX,m_DimensionY;
     G4double Coll_FDx,Coll_FDy;
     G4double Trap_Dx,Trap_Dy,Trap_OffsetX,Trap_OffsetY1,Trap_OffsetY2;
     G4double Trap_Dz,Trap_Dy1,Trap_Dy2,Trap_Dx1,Trap_Dx2,Trap_Dx3,Trap_Dx4;
     G4int    Trap_Nx,Trap_Ny;


     GatePVParameterisation*   m_parameterisation;   //!< PV parameterisation
};

#endif
