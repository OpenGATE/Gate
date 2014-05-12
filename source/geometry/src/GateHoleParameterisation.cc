/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateHoleParameterisation.hh"
#include "G4Trap.hh"
#include "G4LogicalVolume.hh"
#include "G4VisAttributes.hh"

GateHoleParameterisation::GateHoleParameterisation(G4double itsFDx,G4double itsFDy,
   G4double itsDx, G4double itsDy,G4int itsNx,G4int itsNy,G4double itsOffsetX,G4double itsOffsetY1,G4double itsOffsetY2,
   G4double itsDz,G4double itsDy1,G4double itsDy2,G4double itsDx1,G4double itsDx2, G4double itsDx3, G4double itsDx4)
   : GatePVParameterisation(),m_Nx(itsNx),m_Ny(itsNy),m_FDx (itsFDx),m_FDy (itsFDy),m_Dx (itsDx),m_Dy (itsDy),
    m_OffsetX (itsOffsetX),m_OffsetY1 (itsOffsetY1),m_OffsetY2 (itsOffsetY2),m_Dz(itsDz),m_Dy1(itsDy1),m_Dy2(itsDy2),
    m_Dx1(itsDx1),m_Dx2(itsDx2),m_Dx3(itsDx3),m_Dx4(itsDx4)
{
  PreComputeConsts ();
}

void GateHoleParameterisation::ComputeDimensions(G4Trap& Coll_Trap, const G4int copyNo, const G4VPhysicalVolume* ) const
{                                    
  G4int    tmp = ( ( copyNo / m_Nx ) + ( copyNo % m_Nx ) ) % 2;
  
  G4double m_Theta = 0.0;
  G4double m_Phi = 0.0;

  G4double tmp1 = m_Dx1 + tmp * (m_Dx2 - m_Dx1);
  G4double tmp2 = m_Dx2 - tmp * (m_Dx2 - m_Dx1);
  G4double tmp3 = m_Dx3 + tmp * (m_Dx4 - m_Dx3);
  G4double tmp4 = m_Dx4 - tmp * (m_Dx4 - m_Dx3);
  
  if (m_FDx != 0.0)
    {
     m_Theta = atan ( ( m_OffsetX + ( copyNo % m_Nx ) * m_Dx ) / m_FDx );
//     G4cout << "ok\n";
    }

  if (m_FDy != 0.0)
    {
     m_Phi = atan ( ( 0.5 * ( m_OffsetY1 + m_OffsetY2 ) + ( ( copyNo / m_Nx )  + tmp - 0.5 ) * m_Dy ) / m_FDy );
//     G4cout << "ok\n";
    }
//  G4cout << "theta      " << m_Theta << "\n";
//  G4cout << "phi        " << m_Phi << "\n";

  Coll_Trap.SetAllParameters(m_Dz, m_Theta, m_Phi, m_Dy1, tmp1, tmp2, 0., m_Dy2, tmp3, tmp4, 0.);
}

void GateHoleParameterisation::ComputeTransformation(G4int copyNumber, G4VPhysicalVolume *aVolume) const
{
  G4int    tmp = ( ( copyNumber / m_Nx ) + ( copyNumber % m_Nx ) ) % 2;
  G4double x = m_OffsetX + (copyNumber % m_Nx) * m_Dx;
  G4double y = m_OffsetY1 + tmp * (m_OffsetY2 - m_OffsetY1) + (copyNumber / m_Nx) * m_Dy ;
  aVolume->SetTranslation(G4ThreeVector(x,y,0));
}

void GateHoleParameterisation::PreComputeConsts()
{
  m_N = m_Nx * m_Ny;
}


void GateHoleParameterisation::Update(G4double fdX,G4double fdY,G4double dX,G4double dY,
	G4int nX,G4int nY,G4double offsetX,G4double offsetY1,G4double offsetY2,G4double dZ,G4double dY1,G4double dY2,
	G4double dX1,G4double dX2,G4double dX3,G4double dX4)
{
  m_FDx     = fdX;
  m_FDy     = fdY;

  m_Dx      = dX;
  m_Dy      = dY;
  m_Nx      = nX;
  m_Ny      = nY;
  m_OffsetX = offsetX;
  m_OffsetY1 = offsetY1;
  m_OffsetY2 = offsetY2;

  m_Dz      = dZ;
  m_Dy1     = dY1;
  m_Dy2     = dY2;
  m_Dx1     = dX1;
  m_Dx2     = dX2;
  m_Dx3     = dX3;
  m_Dx4     = dX4;

  PreComputeConsts ();
}



