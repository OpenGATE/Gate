/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/* ----------------------------------------------------------------------------- *
 *                                                                         *
 *  Class Description :                                                    *
 *                                                                         *  
 *  To generate the direction of a primary vertex according to the defined *
 *  distribution                                                           *
 *                                                                         * 
 * ----------------------------------------------------------------------------- */ 

#ifndef GateSPSAngDistribution_h
#define GateSPSAngDistribution_h 1

#include "G4SPSAngDistribution.hh"

class GateSPSAngDistribution : public G4SPSAngDistribution
{

 public :
 
  GateSPSAngDistribution () ;
  ~GateSPSAngDistribution () ; 

  // FocusPointCopy is a copy of the private member FocusPoint in G4SPSAngDistribution
  // that cannot be accessed from get because there is no GetFocusPoint and the member
  // is private. Every call to GateSPSAngDistribution::SetFocusPoint is doubled with a
  // call to GateSPSAngDistribution::SetFocusPointCopy to be able to access its value.
  G4ThreeVector GetFocusPointCopy();  
  void 	SetFocusPointCopy (G4ThreeVector);

 private :
  G4ThreeVector FocusPointCopy;
} ;

#endif
