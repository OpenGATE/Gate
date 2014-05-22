/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*----------------------

   GATE - Geant4 Application for Tomographic Emission 
   OpenGATE Collaboration 
     
   Richard Taschereau <rtaschereau@mednet.ucla.edu>
   
   Copyright (C) 2005 Crump Institute for Molecular Imaging, UCLA

This software is distributed under the terms 
of the GNU Lesser General  Public Licence (LGPL) 
See GATE/LICENSE.txt for further details 
----------------------*/


#ifndef GateUtilityForG4ThreeVector_H
#define GateUtilityForG4ThreeVector_H 1

#include "G4ThreeVector.hh"


// Utility methods for G4ThreeVector

inline G4ThreeVector KroneckerProduct(const G4ThreeVector& a, const G4ThreeVector& b){
  return G4ThreeVector(a.x()*b.x(), a.y()*b.y(), a.z()*b.z() );
}

#endif
