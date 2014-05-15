/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/



#ifndef GateInverseSquareBlurringLaw_h
#define GateInverseSquareBlurringLaw_h 1

#include "GateVBlurringLaw.hh"
#include "GateInverseSquareBlurringLawMessenger.hh"

#include "GateTools.hh"
#include "G4UnitsTable.hh"

/*! \class  GateInverseSquareBlurringLaw
    \brief  InverseSquare law giving the resolution in energy


      \sa GateVBluringLaw
*/

class GateInverseSquareBlurringLaw  : public GateVBlurringLaw {

	public :
		GateInverseSquareBlurringLaw(const G4String& itsName, G4double itsReferenceEnergy=-1., G4double itsResolution=0.);
    	virtual ~GateInverseSquareBlurringLaw() {delete m_messenger;}

    	// Implementation of the pure virtual method in GateVBlurringLaw
    	// Inverse Square resolution is calculated by  R = R0*sqrt(E0)/sqrt(E)
       	virtual G4double ComputeResolution(G4double energy) const;

       	inline G4double GetResolution() const { return m_resolution; }
       	inline G4double GetEnergyRef() const { return m_eref; }

       	inline void SetResolution(G4double res) { m_resolution = res; }
       	inline void SetEnergyRef(G4double ener) { m_eref = ener; }


       	// Implementation of the virtual method in GateVBlurringLaw
       	// Also called by GateBlurring
       	virtual void DescribeMyself (size_t ident=0) const;



    private :
    	G4double m_eref;
    	G4double m_resolution;
    	GateInverseSquareBlurringLawMessenger* m_messenger;

};

#endif
