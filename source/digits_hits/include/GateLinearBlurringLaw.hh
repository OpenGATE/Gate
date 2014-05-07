/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GateLinearBlurringLaw_h
#define GateLinearBlurringLaw_h 1

#include "GateVBlurringLaw.hh"
#include "GateLinearBlurringLawMessenger.hh"

#include "GateTools.hh"
#include "G4UnitsTable.hh"

/*! \class  GateLinearBlurringLaw
    \brief  Linear law giving the resolution in energy


      \sa GateVBluringLaw
*/

class GateLinearBlurringLaw  : public GateVBlurringLaw {

	public :
		GateLinearBlurringLaw(const G4String& itsName, G4double itsReferenceEnergy=-1., G4double itsResolution=0., G4double itsSlope=0.);
    	virtual ~GateLinearBlurringLaw() { delete m_messenger; }

    	// Implementation of the pure virtual method of the GateVBlurringLaw class
    	// Linear resolution is calculated by  R = s(E-E0) + R0
       	virtual G4double ComputeResolution(G4double energy) const;

       	inline G4double GetResolution() const { return m_resolution; }
       	inline G4double GetEnergyRef() const { return m_eref; }
       	inline G4double GetSlope() const { return m_slope; }

       	inline void SetResolution(G4double res) { m_resolution = res; }
       	inline void SetEnergyRef(G4double ener) { m_eref = ener; }
       	inline void SetSlope(G4double slope) { m_slope = slope; }

       	// Implementation of the pure virtual method of the GateVBlurringLaw class
       	// Also called by GateBlurring
       	virtual void DescribeMyself (size_t indent=0) const;

    private :
    	G4double m_eref;
    	G4double m_resolution;
    	G4double m_slope;
    	GateLinearBlurringLawMessenger* m_messenger;


};

#endif
