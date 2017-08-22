/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateScintillation_h
#define GateScintillation_h 1

/*!
 * \file GateScintillation.h
 * \author Mathieu Dupont <mdupont@cppm.in2p3.fr>
 *
 */

#include "G4Scintillation.hh"

/*!
 * \class GateScintillation
 *
 * With this class, Gate is able to use FiniteRiseTime for scintillation process.
 * Before call to G4Scintillation::PostStepDoIt, GateScintillation class check existence of FASTSCINTILLATIONRISETIME
 * or SLOWSCINTILLATIONRISETIME in MaterialPropertiesTable for the current G4Material
 * and set the FiniteRiseTime flag of G4Scintillation consequently.
 *
 *
 */

class GateScintillation : public G4Scintillation
{

  using G4Scintillation::G4Scintillation;

	G4VParticleChange* PostStepDoIt(const G4Track& aTrack, 
			                const G4Step&  aStep);

};


#endif /* GateScintillation_h */
