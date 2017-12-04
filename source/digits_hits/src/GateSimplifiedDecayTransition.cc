/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateSimplifiedDecayTransition.hh"
#include <G4PhysicalConstants.hh>
#include <cmath>

// A few static constants
double GateSimplifiedDecayTransition::kAlphaSquared = fine_structure_const*fine_structure_const;

//----------------------------------------------------------------------------------------
// Fermi Function:
//     given eKin, eMax and Z, it returns the normalized probability density
//     of emitting a positron with energy eKin
//
//     eKin is in units of MeV/mec2 (0.511 MeV)

double GateSimplifiedDecayTransition::fermiFunction(double eKin){

  double deltaE        ( energy/electron_mass_c2 - eKin );
  double deltaESquared ( deltaE * deltaE );
  double X             ( eKin + 1);
  double X2            ( X*X     );
  double Zeff          ( 1 - atomicNumber   );
  double ZeffSquared   ( Zeff*Zeff   );

  return normalisationFactor * 2*deltaESquared*fine_structure_const*pi*X2*Zeff*
    pow((-1 + X2)/4. + kAlphaSquared*X2*ZeffSquared, -1 + sqrt(1 - kAlphaSquared*ZeffSquared))
    / (1 - exp(-2*fine_structure_const*pi*X*Zeff/sqrt(-1 + X2))  );

}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
double GateSimplifiedDecayTransition::simpleHitAndMiss(){
  double x( energy    * G4UniformRand() );
  double y( amplitude * G4UniformRand() );

  while(  y >  fermiFunction(x/electron_mass_c2)  ) {
    x = energy    * G4UniformRand();
    y = amplitude * G4UniformRand();
  }
  return x;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
double GateSimplifiedDecayTransition::majoredHitAndMiss(){
  double xi( CDFRandom() );
  double yi( G4UniformRand() * majoringFunction(xi) );

  while(  yi >  fermiFunction(xi/electron_mass_c2)  ){
    xi = CDFRandom();
    yi = G4UniformRand() * majoringFunction(xi);
  }
  return xi;
}
//----------------------------------------------------------------------------------------
