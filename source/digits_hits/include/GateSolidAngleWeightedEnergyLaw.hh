/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#ifndef GateSolidAngleWeightedEnergyLaw_h
#define GateSolidAngleWeightedEnergyLaw_h 1

#include "GateVEffectiveEnergyLaw.hh"
#include "GateSolidAngleWeightedEnergyLawMessenger.hh"

#include "GateTools.hh"
#include "G4UnitsTable.hh"
#include "G4VoxelLimits.hh"


class GateSolidAngleWeightedEnergyLaw  : public GateVEffectiveEnergyLaw {

	public :
        GateSolidAngleWeightedEnergyLaw(const G4String& itsName, G4double itsRectangleSzX=-1., G4double itsRectangleSzY=-1.);
        virtual ~GateSolidAngleWeightedEnergyLaw() {delete m_messenger;}


        virtual G4double ComputeEffectiveEnergy(GateDigi digi) const;

        inline G4double GetRectangleSzX() const { return m_szX; }
        inline G4double GetRectangleSzY() const { return m_szY; }
        inline G4double GetZSense() const { return  m_zSense4Readout; }

        inline void SetRectangleSzX(G4double SzX) { m_szX = SzX; }
        inline void SetRectangleSzY(G4double SzY) { m_szY = SzY; }
         inline void SetZSense(G4int senseZ) { m_zSense4Readout= senseZ; }



       	virtual void DescribeMyself (size_t ident=0) const;



    private :
        G4double m_szX;
        G4double m_szY;
        G4int  m_zSense4Readout;
        GateSolidAngleWeightedEnergyLawMessenger* m_messenger;

};

#endif
