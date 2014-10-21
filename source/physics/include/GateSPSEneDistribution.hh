/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/* ---------------------------------------------------------------------------- *
 *                                                                              *
 *  Class Description :                                                         *
 *                                                                              *  
 *  To generate the energy of a primary vertex according to the defined         *
 *  distribution                                                                *
 *                                                                              *
 *   Revision 1.5 2014/08/1 Yann PERROT and Simon NICOLAS  LPC Clermont-ferrand *
 *   Solution for generating particles from Energy spectra (discrete spectrum,  *
 *   histogram and linear interpolation)                                        *  
 *   Creation of two new methods:                                               *      
 *     ConstructUserSpectrum() and GenerateFromUserSpectrum()                   *                                                                
 * -----------------------------------------------------------------------------*/ 

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

  // Create probability tables
  void GenerateFromUserSpectrum(); 
  // Shoot an energy in previously created probability tables
  void BuildUserSpectrum(G4String FileName);
    
  void PrintMessage();
  G4double GenerateOne( G4ParticleDefinition* ) ;
  inline void SetMinEnergy( G4double E ){ m_Emin = E;};
  inline void SetEnergyRange( G4double r ){ m_EnergyRange = r; };
 private :
 
  G4double particle_energy ;
  G4double      m_Emin;
  G4double      m_EnergyRange;
  
  G4int      m_mode;
  G4int      m_dim_spectrum;
  G4double   m_sum_proba;
  G4double*  m_tab_proba;
  G4double*  m_tab_sumproba;
  G4double*  m_tab_energy;

} ;

#endif
