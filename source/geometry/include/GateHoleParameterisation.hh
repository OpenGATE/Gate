/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateHoleParameterisation_H
#define GateHoleParameterisation_H 1

#include "globals.hh"

#include "GatePVParameterisation.hh"

class G4VPhysicalVolume;
class G4Trap;

class GateHoleParameterisation : public GatePVParameterisation
{
   public:
      GateHoleParameterisation(G4double itsFDx,G4double itsFDy,G4double itsDx,G4double itsDy,
         G4int itsNx,G4int itsNy,G4double itsOffsetX,G4double itsOffsetY1,G4double itsOffsetY2,
	 G4double itsDz, G4double itsDy1, G4double itsDy2, G4double itsDx1,
         G4double itsDx2, G4double itsDx3, G4double itsDx4);
   
   virtual ~GateHoleParameterisation() {}

   virtual void ComputeTransformation(const G4int copyNumber, G4VPhysicalVolume *aVolume) const;

   using G4VPVParameterisation::ComputeDimensions;
   virtual void ComputeDimensions(G4Trap& Coll_Trap, const G4int copyNo, const G4VPhysicalVolume* physVol) const;

   virtual inline int GetNbOfCopies()
      { return m_N; }

   void Update(G4double fdX,G4double fdY,G4double dX,G4double dY,G4int nX,G4int nY,G4double offsetX,G4double offsetY1,
   	G4double offsetY2,G4double dZ,G4double dY1,G4double dY2,G4double dX1,G4double dX2,G4double dX3,G4double dX4);
	
   void PreComputeConsts();

   protected:

      G4int m_Nx,m_Ny,m_N;
      G4double m_FDx,m_FDy,m_Dx,m_Dy,m_OffsetX,m_OffsetY1,m_OffsetY2;
      G4double m_Dz,m_Dy1,m_Dy2,m_Dx1,m_Dx2,m_Dx3,m_Dx4;

};
#endif

