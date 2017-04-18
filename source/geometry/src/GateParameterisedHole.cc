/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateParameterisedHole.hh"

#include "G4VPhysicalVolume.hh"
#include "G4PVParameterised.hh"

#include "G4UnitsTable.hh"

#include "GateTools.hh"

GateParameterisedHole::GateParameterisedHole(const G4String& itsName,const G4String& itsMaterialName,
     G4double itsFocalDistanceX,G4double itsFocalDistanceY,G4double itsSeptalThickness,G4double itsInnerRadius,
     G4double itsHeight,G4double itsDimensionX,G4double itsDimensionY)
     : GateTrap(itsName,itsMaterialName,1.,0.,0.,1.,1.,1.,0.,1.,1.,1.,0.),
     m_FocalDistanceX (itsFocalDistanceX),m_FocalDistanceY (itsFocalDistanceY),m_SeptalThickness (itsSeptalThickness),
     m_InnerRadius (itsInnerRadius),m_Height (itsHeight),m_DimensionX (itsDimensionX),m_DimensionY (itsDimensionY)
{
  PreComputeConstants ();
  SetMaterialName(itsMaterialName);
  SetParameterisation ( new GateHoleParameterisation(Coll_FDx,Coll_FDy,Trap_Dx,Trap_Dy,Trap_Nx,Trap_Ny,
  	Trap_OffsetX,Trap_OffsetY1,Trap_OffsetY2,Trap_Dz,Trap_Dy1,Trap_Dy2,Trap_Dx1,Trap_Dx2,Trap_Dx3,Trap_Dx4) );
}

GateParameterisedHole::~GateParameterisedHole()
{

  if (m_parameterisation)
    delete m_parameterisation;
    
}

void GateParameterisedHole::PreComputeConstants()
{
   Coll_FDx = m_FocalDistanceX;
   Coll_FDy = m_FocalDistanceY;

   Trap_Dz  = 0.5 * m_Height;
   Trap_Dy  = m_InnerRadius + m_SeptalThickness;
//   Trap_Dx  = (4.0 * m_InnerRadius + 2.0 * ( m_InnerRadius + m_SeptalThickness ) ) / sqrt(3.0);
   Trap_Dx  = ( 2.0 * m_InnerRadius + ( m_InnerRadius + m_SeptalThickness ) ) / sqrt(3.0);

   if (m_FocalDistanceY != 0.0)
      {
       Trap_Dy1 = 0.5 * m_InnerRadius ;//* m_FocalDistanceY / ( m_FocalDistanceY + Trap_Dz ) ;
       Trap_Dy2 = 0.5 * m_InnerRadius ;// ( m_FocalDistanceY + m_Height ) / ( m_FocalDistanceY + Trap_Dz ) ;
       Trap_Ny  = int ( floor ( m_DimensionY * ( m_FocalDistanceY + Trap_Dz ) / ( m_FocalDistanceY + m_Height ) / Trap_Dy ) );
      }
   else
      {
       Trap_Dy1 = 0.5 * m_InnerRadius;
       Trap_Dy2 = 0.5 * m_InnerRadius;
       Trap_Ny  = int ( floor ( m_DimensionY / Trap_Dy ) );
       G4cout << "ok\n";
      }

   Trap_OffsetY1 = - 0.5 * ( Trap_Ny - 1.0 ) * Trap_Dy - 0.5 * m_SeptalThickness;
   Trap_OffsetY2 = - 0.5 * ( Trap_Ny - 1.0 ) * Trap_Dy + 0.5 * m_SeptalThickness;

   if (m_FocalDistanceX != 0.0)
      {
       Trap_Dx1 = m_FocalDistanceX / ( m_FocalDistanceX + Trap_Dz ) * 2.0 * m_InnerRadius / sqrt(3.0);
       Trap_Dx3 = ( m_FocalDistanceX + m_Height ) / ( m_FocalDistanceX + Trap_Dz ) * 2.0 * m_InnerRadius / sqrt(3.0);
       Trap_Dx2 = Trap_Dx1 - m_InnerRadius / sqrt(3.0);
       Trap_Dx4 = Trap_Dx3 - m_InnerRadius / sqrt(3.0);
       Trap_Nx  = int ( floor ( m_DimensionX * ( m_FocalDistanceX + Trap_Dz ) / ( m_FocalDistanceX + m_Height ) / Trap_Dx ) );
      }
   else
      {
       Trap_Dx1 = 2.0 * m_InnerRadius / sqrt(3.0);
       Trap_Dx3 = 2.0 * m_InnerRadius / sqrt(3.0);
       Trap_Dx2 = Trap_Dx1 - m_InnerRadius / sqrt(3.0);
       Trap_Dx4 = Trap_Dx3 - m_InnerRadius / sqrt(3.0);
       Trap_Nx  = int ( floor ( m_DimensionX / Trap_Dx ) );
       G4cout << "ok\n";
      }

   Trap_OffsetX = - 0.5 * ( Trap_Nx - 1.0 ) * Trap_Dx;

   G4cout << "Nx      " << Trap_Nx << "\n";
   G4cout << "Ny      " << Trap_Ny << "\n";
   G4cout << "Dy      " << Trap_Dy << "\n";
   G4cout << "Dx      " << Trap_Dx << "\n";
   G4cout << "OffsetX " << Trap_OffsetX << "\n";
   G4cout << "OffsetY " << Trap_OffsetY1 << "\n";
}

void GateParameterisedHole::ResizeHole(G4double FocalDistanceX,G4double FocalDistanceY,G4double SeptalThickness,
      G4double InnerRadius,G4double Height,G4double DimensionX,G4double DimensionY)
{

   m_FocalDistanceX  = FocalDistanceX;
   m_FocalDistanceY  = FocalDistanceY;
   m_SeptalThickness = SeptalThickness;
   m_InnerRadius     = InnerRadius;
   m_Height          = Height;
   m_DimensionX      = DimensionX;
   m_DimensionY      = DimensionY;

   PreComputeConstants ();

   GetHoleCreator()->SetTrapDy1 (Trap_Dy1);
   GetHoleCreator()->SetTrapDy2 (Trap_Dy2);
   GetHoleCreator()->SetTrapDx1 (Trap_Dx1);
   GetHoleCreator()->SetTrapDx2 (Trap_Dx2);
   GetHoleCreator()->SetTrapDx3 (Trap_Dx3);
   GetHoleCreator()->SetTrapDx4 (Trap_Dx4);

   GetHoleParameterisation()->Update(Coll_FDx,Coll_FDy,Trap_Dx,Trap_Dy,Trap_Nx,Trap_Ny,
    Trap_OffsetX,Trap_OffsetY1,Trap_OffsetY2,Trap_Dz,Trap_Dy1,Trap_Dy2,Trap_Dx1,Trap_Dx2,Trap_Dx3,Trap_Dx4);
}



// Implementation of the pure virtual method declared by the base class GateVCreator
// If flagUpdateOnly is 0, it creates a new parameterised, using the parameterisation
// This parameterisation must have been created by a concrete class derived from GateVParameterised
// If flagUpdateOnly is set to 1, the default position is updated 
void GateParameterisedHole::ConstructOwnPhysicalVolume(G4bool flagUpdateOnly)
{
  // In build mode, we must create a new parameterised
  // In update mode, there's nothing to do (that's cool!)
  if (!flagUpdateOnly) {
    // Create the parameterised
    G4VPhysicalVolume *physicalVolume
      	    = new G4PVParameterised(mPhysicalVolumeName,       
                              	    GetCreator()->GetLogicalVolume(),    
                              	    pMotherLogicalVolume,    
				    kZAxis,
                              	    m_parameterisation->GetNbOfCopies(), 
                              	    m_parameterisation); 
    // Store it into the physical volume vector
    PushPhysicalVolume(physicalVolume);
  }
}
