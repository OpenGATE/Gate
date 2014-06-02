/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateInverseSquareBlurringLaw.hh"

/*! \class  GateInverseSquareBlurringLaw
  \brief  InverseSquare law giving the resolution in energy
  \sa GateVBluringLaw
*/


GateInverseSquareBlurringLaw::GateInverseSquareBlurringLaw(const G4String& itsName, G4double itsReferenceEnergy, G4double itsResolution) :
	GateVBlurringLaw(itsName),
	m_eref(itsReferenceEnergy),
	m_resolution(itsResolution)
{
	m_messenger = new GateInverseSquareBlurringLawMessenger(this);
}



G4double GateInverseSquareBlurringLaw::ComputeResolution(G4double energy) const {

	if(m_resolution < 0. ) {
		G4cerr << 	G4endl << "[GateInverseSquareBlurringLaw::ComputeResolution]:" << G4endl
      	   <<   "Sorry, but the resolution (" << GetResolution() << ") is invalid" << G4endl;
    	G4Exception( "GateInverseSquareBlurringLaw::ComputeResolution", "ComputeResolution", FatalException, "You must set the energy of reference AND the resolution:\n\t/gate/digitizer/blurring/inverseSquare/setResolution NUMBER\n or disable the blurring using:\n\t/gate/digitizer/blurring/disable\n");
	}
	else if (m_eref < 0.) {
		G4cerr <<   G4endl << "[GateInverseSquareBlurringLaw::ComputeResolution]:" << G4endl
			<<   "Sorry, but the energy of reference (" << G4BestUnit(GetEnergyRef(),"Energy") << ") is invalid" << G4endl;
	 	G4Exception( "GateInverseSquareBlurringLaw::ComputeResolution", "ComputeResolution", FatalException, "You must set the resolution AND the energy of reference:\n\t/gate/digitizer/blurring/inverseSquare/setEnergyOfReference ENERGY\n or disable the blurring using:\n\t/gate/digitizer/blurring/disable\n");
	}

	return ((m_resolution * sqrt(m_eref)) / sqrt(energy));

}

void GateInverseSquareBlurringLaw::DescribeMyself (size_t indent) const {
	G4cout << "Inverse Square law for energy blurring" << G4endl;
	G4cout << GateTools::Indent(indent) << "Energy of Reference:\t" << G4BestUnit(GetEnergyRef(),"Energy") << G4endl;
	G4cout << GateTools::Indent(indent) << "Resolution of Reference:\t" << GetResolution() << G4endl;
}
