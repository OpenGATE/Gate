/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
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
 * 2020/10/21: Implementation of IsApplicable has been changed between geant4 10.05 and geant4 10.06 which
 * made Scintillation not applicable for gamma. Because gamma has a PDGCharge of 0.0. Here we go back to previous
 * implementation until we find another solution.
 *
 *
 */

class GateScintillation : public G4Scintillation
{

  using G4Scintillation::G4Scintillation;

	G4VParticleChange* PostStepDoIt(const G4Track& aTrack, 
			                const G4Step&  aStep) override ;

  G4bool IsApplicable(
      const G4ParticleDefinition& aParticleType) override;



};


#endif /* GateScintillation_h */
