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
 *  To generate the energy of a primary vertex according to the defined    *
 *  distribution                                                           *
 *                                                                         * 
 * ----------------------------------------------------------------------------- */ 

#ifndef GateSPSEneDistribution_h
#define GateSPSEneDistribution_h 1

#include "G4PhysicsOrderedFreeVector.hh"
#include "G4ParticleMomentum.hh"
#include "G4ParticleDefinition.hh"
#include "G4DataInterpolation.hh"

#include "G4SPSEneDistribution.hh"

class GateSPSEneDistribution : public G4SPSEneDistribution
{

 public :
 
  GateSPSEneDistribution () ;
  ~GateSPSEneDistribution () ; 
  
  void GenerateFluor18() ;
  void GenerateOxygen15() ;
  void GenerateCarbon11() ; 
  void GenerateRangeEnergy();
  G4double GenerateOne( G4ParticleDefinition* ) ;
  inline void SetMinEnergy( G4double E ){ m_Emin = E;};
  inline void SetEnergyRange( G4double r ){ m_EnergyRange = r; };
 private :
 
  G4double particle_energy ;
  G4double      m_Emin;
  G4double      m_EnergyRange;
  
} ;

#endif
