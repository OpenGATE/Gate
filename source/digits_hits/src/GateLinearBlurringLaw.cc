/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "G4SystemOfUnits.hh"

#include "GateLinearBlurringLaw.hh"

/*! \class  GateLinearBlurringLaw
  \brief  Linear law giving the resolution in energy
  \sa GateVBluringLaw
*/


GateLinearBlurringLaw::GateLinearBlurringLaw(const G4String& itsName, G4double itsReferenceEnergy, G4double itsResolution, G4double itsSlope) :
	GateVBlurringLaw(itsName),
	m_eref(itsReferenceEnergy),
	m_resolution(itsResolution),
	m_slope(itsSlope)
{
	 new G4UnitDefinition ( "1/electronvolt", "1/eV", "Energy Slope", 1/electronvolt );
	 new G4UnitDefinition ( "1/kiloelectronvolt", "1/keV", "Energy Slope", 1/kiloelectronvolt );
	 new G4UnitDefinition ( "1/megaelectronvolt", "1/MeV", "Energy Slope", 1/megaelectronvolt );
	 new G4UnitDefinition ( "1/gigaelectronvolt", "1/GeV", "Energy Slope", 1/gigaelectronvolt );
	 new G4UnitDefinition ( "1/joule", "1/J", "Energy Slope", 1/joule );

	m_messenger = new GateLinearBlurringLawMessenger(this);
}


G4double GateLinearBlurringLaw::ComputeResolution(G4double energy) const {

	if(m_resolution < 0. ) {
		G4cerr << 	G4endl << "[GateLinearBlurringLaw::ComputeResolution]:" << G4endl
      	   <<   "Sorry, but the resolution (" << GetResolution() << ") is invalid" << G4endl;
    	G4String msg = "You must set the energy of reference AND the resolution AND the slope:\n"
      "\t/gate/digitizer/blurring/linear/setResolution NUMBER\n"
      "or disable the blurring using:\n"
      "\t/gate/digitizer/blurring/disable\n";

			G4Exception( "GateLinearBlurringLaw::ComputeResolution", "ComputeResolution", FatalException, msg );
	}
	else if (m_eref < 0.) {
		G4cerr <<   G4endl << "[GateLinearBlurringLaw::ComputeResolution]:" << G4endl
			<<   "Sorry, but the energy of reference (" << G4BestUnit(GetEnergyRef(),"Energy") << ") is invalid" << G4endl;

		G4String msg = "You must set the resolution AND the energy of reference AND the slope:\n"
         "\t/gate/digitizer/blurring/linear/setEnergyOfReference ENERGY\n"
         "or disable the blurring using:\n"
         "\t/gate/digitizer/blurring/disable\n";

	 	G4Exception( "GateLinearBlurringLaw::ComputeResolution", "ComputeResolution", FatalException, msg );
	}

	return (m_slope * (energy - m_eref) + m_resolution) ;
}


void GateLinearBlurringLaw::DescribeMyself (size_t indent) const {
	G4cout << "Linear law for energy blurring" << G4endl;
	G4cout << GateTools::Indent(indent) << "Energy of Reference:\t" << G4BestUnit(GetEnergyRef(),"Energy") << G4endl;
	G4cout << GateTools::Indent(indent) << "Resolution of Reference:\t" << GetResolution() << G4endl;
	G4cout << GateTools::Indent(indent) << "Slope:\t" << G4BestUnit(GetSlope(),"Energy Slope") << G4endl;
}
