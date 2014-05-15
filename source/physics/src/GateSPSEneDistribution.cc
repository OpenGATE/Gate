/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "Randomize.hh"

#include "GateSPSEneDistribution.hh"


GateSPSEneDistribution::GateSPSEneDistribution()
{

}

GateSPSEneDistribution::~GateSPSEneDistribution()
{

}
 
void GateSPSEneDistribution::GenerateFluor18()
{ 
  // Fit parameters for the Fluor18 spectra
  G4double a = 10.2088 ;
  G4double b = -30.4551 ;
  G4double c = 28.4376 ;
  G4double d = -7.9828 ;
  G4double E ; 
  G4double u ; 
  G4double energyF18 = 0. ;
  
   do
    {
     E = 0.511 + ( 1.144 - 0.511 ) * G4UniformRand() ; // Emin = 0.511 ; Emax = 1.144
     u = 0.5209 * G4UniformRand() ;   // Nmin = 0 ; Nmax = 0.5209
     energyF18 = E;  
    }
   while ( u > a*E*E*E + b*E*E + c*E + d ) ; 
  G4double energy_fluor = energyF18 - 0.511 ;  
  particle_energy = energy_fluor ;
  
}

void GateSPSEneDistribution::GenerateOxygen15()
{ 
  // Fit parameters for the Oxygen15 spectra 
  G4double a = 3.43874 ;
  G4double b = -9.04016 ;
  G4double c = -7.71579 ;
  G4double d = 13.3147 ;
  G4double e = 32.5321 ;
  G4double f = -18.8379 ;
  G4double E ;
  G4double u ;
  G4double energyO15 = 0. ;
   do
    {
     E = CLHEP::RandFlat::shoot( 0.511, 2.249 ) ; // Emin ; Emax
     u = CLHEP::RandFlat::shoot( 0., 15.88 ) ;   // Nmin ; Nmax
     energyO15 = E ;  
    }
   while ( u > a*E*E*E*E*E + b*E*E*E*E + c*E*E*E + d*E*E + e*E + f ) ; 
  G4double energy_oxygen = energyO15 - 0.511 ; 
  particle_energy = energy_oxygen ;
  
}

void GateSPSEneDistribution::GenerateCarbon11()
{ 
  // Fit parameters for the Carbon11 spectra  
  G4double a = 2.36384 ;
  G4double b = -1.00671 ;
  G4double c = -7.07171 ;
  G4double d = -7.84014 ;
  G4double e = 26.0449 ;
  G4double f = -10.4374 ;
  G4double E ;
  G4double u ; 
  G4double energyC11 = 0. ;
   do
    {
     E = CLHEP::RandFlat::shoot( 0.511, 1.47 ) ; // Emin ; Emax
     u = CLHEP::RandFlat::shoot( 0., 2.2 ) ;   // Nmin ; Nmax
     energyC11 = E;  
    }
   while ( u > a*E*E*E*E*E + b*E*E*E*E + c*E*E*E + d*E*E + e*E + f ) ; 
  G4double energy_carbon = energyC11 - 0.511 ;  
  particle_energy = energy_carbon ;
}

G4double GateSPSEneDistribution::GenerateOne( G4ParticleDefinition* a )
{  
     
  if( GetEnergyDisType() == "Fluor18" )
   GenerateFluor18();
  else if( GetEnergyDisType() == "Oxygen15" )
   GenerateOxygen15();
  else if( GetEnergyDisType() == "Carbon11" )
   GenerateCarbon11();
  else if(GetEnergyDisType() == "Range")
    GenerateRangeEnergy();
	else  
   particle_energy = G4SPSEneDistribution::GenerateOne( a ) ;   
         
 return particle_energy ;

}
//PY.Descourt 08/09/2009
void GateSPSEneDistribution::GenerateRangeEnergy()
{

particle_energy = (m_Emin  + G4UniformRand() * m_EnergyRange) ;

//G4cout << " GateSource::GenerateEnergyFromRange()  energy " << Energy<<G4endl;

}

