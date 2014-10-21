/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "Randomize.hh"

#include "GateSPSEneDistribution.hh"
#include <fstream>
#include "G4SystemOfUnits.hh"
using namespace std;

GateSPSEneDistribution::GateSPSEneDistribution():
  G4SPSEneDistribution()
{
 
}

GateSPSEneDistribution::~GateSPSEneDistribution()
{
  delete[] m_tab_sumproba;
  delete[] m_tab_energy;
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
  else if(GetEnergyDisType() == "UserSpectrum") 
    GenerateFromUserSpectrum();
	else particle_energy = G4SPSEneDistribution::GenerateOne( a );   
    
 return particle_energy ;

}
//PY.Descourt 08/09/2009
void GateSPSEneDistribution::GenerateRangeEnergy()
{

particle_energy = (m_Emin  + G4UniformRand() * m_EnergyRange) ;


}

// BuildSpectrum fonction read a file which contain in first line mode of spectrum (Spectrum line (1), Histogramm (2)
// or linear interpolated spectrum (3)) in first and in second start energy for the spectrum. Start energy it used only 
// for histogram mode but you want give default value (0 for example), even if this value don't serve for mode (1) and (3)  
// After the first line each line contain in first column energy and in second probability for this energy.
void GateSPSEneDistribution::BuildUserSpectrum(G4String FileName)
{
  
  ifstream inputFile (FileName.data());
  G4int nline(0);
  if(inputFile) {
    G4String line;
    G4double proba_read;
    G4double energy_read;

    inputFile >> m_mode ;
    inputFile >> m_Emin ;

    G4int cursor_position = inputFile.tellg(); // tellg() save file cursor position

    while(getline(inputFile, line)) nline++; // count number of line of inputFile 
    m_dim_spectrum = nline-1;

    

    // create two tables for energy and probability 
    m_tab_energy= new G4double[m_dim_spectrum];
    m_tab_proba= new G4double[m_dim_spectrum];
  
    nline = 0;

    inputFile.clear();
    inputFile.seekg(cursor_position, inputFile.beg); // return to the 2nd line in the file 

     while(nline<m_dim_spectrum){
      inputFile >> energy_read;
      inputFile >> proba_read;

      m_tab_energy[nline]=energy_read;
      m_tab_proba[nline]=proba_read;
      nline++;
    }


    inputFile.close();  
     

    // Construct probability table
    m_sum_proba=0;
    nline=0;

    switch(m_mode){
      case 1: // probability table to create discrete spectrum
        G4cout << "Discrete spectrum" << G4endl;
        m_tab_sumproba=new G4double[m_dim_spectrum];
        while(nline<m_dim_spectrum){
          m_sum_proba=m_sum_proba+m_tab_proba[nline]; 
          m_tab_sumproba[nline]=m_sum_proba; 
          nline++;
        }
        PrintMessage();
        break;
      case 2: // probability table to create histogram
        G4cout << "Histogram spectrum" << G4endl;
        m_tab_sumproba=new G4double[m_dim_spectrum];
        m_sum_proba=m_tab_proba[0]*(m_tab_energy[0]-m_Emin);
        m_tab_sumproba[0]=m_sum_proba;
        for(nline=1;nline<m_dim_spectrum;nline++){
          m_sum_proba=m_sum_proba+(m_tab_energy[nline]-m_tab_energy[nline-1])*m_tab_proba[nline];
          m_tab_sumproba[nline]=m_sum_proba;
        }
        PrintMessage();
        break;
      case 3: // probability table to create interpolated spectrum
        G4cout << "Interpolated spectrum" << G4endl;
        m_tab_sumproba= new G4double[m_dim_spectrum-1];
        for(nline=1;nline<m_dim_spectrum;nline++){
          m_sum_proba=m_sum_proba+(m_tab_energy[nline]-m_tab_energy[nline-1])*m_tab_proba[nline-1]-0.5*(m_tab_energy[nline]-m_tab_energy[nline-1])*(m_tab_proba[nline-1]-m_tab_proba[nline]);
          m_tab_sumproba[nline-1]=m_sum_proba;
        }
        PrintMessage();
        break;
      default:
      G4Exception("GateSPSEneDistribution::BuildUserSpectrum", "BuildUserSpectrum", FatalException, "Spectrum mode is not recognized, check your spectrum file." );
        break;

    }

  }

  else 
  {
    G4Exception("GateSPSEneDistribution::BuildUserSpectrum", "BuildUserSpectrum", FatalException, "The User Spectrum is not found." );
  }
}



void GateSPSEneDistribution::GenerateFromUserSpectrum()
{  
     
G4int i=0;
G4double pEnergy(0);
G4double my_rndm=G4UniformRand();
while(my_rndm>=(m_tab_sumproba[i])/m_sum_proba) i++; 


G4double a,b,alpha,beta,gamma,X;

switch(m_mode){
  case 1:
    //Discrete Spectrum
    pEnergy= m_tab_energy[i];

    break;
  case 2:
    //Histogram spectrum
    if(i==0) pEnergy=(m_tab_energy[0]-m_Emin)*G4UniformRand()+ m_Emin;
    else pEnergy=(m_tab_energy[i]-m_tab_energy[i-1])*G4UniformRand()+ m_tab_energy[i-1];    
    break;
  case 3:
    //Interpolated spectrum
    a=m_tab_energy[i];
    b=m_tab_energy[i+1]; 

    alpha=(m_tab_proba[i+1]-m_tab_proba[i])/(m_tab_energy[i+1]-m_tab_energy[i]);
    beta=m_tab_proba[i+1]-m_tab_energy[i+1]*alpha;
    gamma=(alpha/2)*(b*b-a*a)+beta*(b-a);
    my_rndm=G4UniformRand();
    X =(-beta+sqrt((alpha*a+beta)*(alpha*a+beta)+2*alpha*gamma*my_rndm))/(alpha);

    if((X-a)*(X-b)<=0) pEnergy=X;
    else pEnergy=(-beta-sqrt((alpha*a+beta)*(alpha*a+beta)+2*alpha*gamma*my_rndm))/(alpha);
    break;
  default:
    pEnergy=0;
    break;

}

    particle_energy = pEnergy;   


}

void GateSPSEneDistribution::PrintMessage() {
  G4cout << "####Energy spectrum correctly uploaded###" << G4endl;
}
 
