/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/* ----------------------------------------------------------------------------- *
 *                                                                         *
 *  Class Description :                                                    *
 *                                                                         *  
 *  To generate the position of a primary vertex according to the defined  *
 *  distribution                                                           *
 *                                                                         * 
 * ----------------------------------------------------------------------------- */

#ifndef GateSPSPosDistribution_h
#define GateSPSPosDistribution_h 1

#include "G4SPSPosDistribution.hh"
#include "G4VPhysicalVolume.hh"
#include <vector>
#include "GateConfiguration.h"

//-------------------------------------------------------------------------------------------------
class GateSPSPosDistribution : public G4SPSPosDistribution
{
  
 public :
 
  GateSPSPosDistribution () ;
  ~GateSPSPosDistribution () ;
  
  void GeneratePositronRange() ;
  void SetPositronRange( G4String ) ;
  
  void ForbidSourceToVolume(const G4String&);
  
  G4ThreeVector GenerateOne() ;
  
  void setVerbosity( G4int );

  inline G4ThreeVector GetPositionVector()
  {return particle_position;}

 private :
  
  G4String positronrange ;
  G4ThreeVector particle_position ;
  
  G4bool IsSourceForbidden();
  G4bool Forbid;
  std::vector<G4VPhysicalVolume*> ForbidVector;
  G4int verbosityLevel;
  
  G4Navigator* gNavigator;
  
} ;
//-------------------------------------------------------------------------------------------------

#endif
