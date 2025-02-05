/*----------------------
   Copyright (C): OpenGATE Collaboration
   
  

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/
/*!
 * GateDistributionTruncatedGuassian (by marc.granado@universite-paris-saclay.fr)
 * 
 * In order to be able to optimise the use of different truncated Gaussians on each event, implying to the use of any FWHM distribution within the crystal,
 * the instance methods: "computeTruncatedSigma" (to correct the sigma in order to preserve the variance of the new distribution)
 * and "ShootTruncatedGaussian" (to generate the random number) have their static method counterpart:
 * "computeTruncatedSigmaStatic" and "ShootRandom" respectively to avoid the need of an object of the class "GateDistributionTruncatedGaussian" to generate the nubmers
 *
 */

#include "GateDistributionTruncatedGaussian.hh"

//#include "GateDistributionTruncatedGaussianMessenger.hh"
#include <math.h>
//#include <CLHEP/Random/RandGauss.h>
#include "Randomize.hh"
#include "GateTools.hh"
#include "GateConstants.hh"


GateDistributionTruncatedGaussian::GateDistributionTruncatedGaussian(const G4String& itsName)
  : GateVDistribution(itsName)
  , m_Mu(0)
  , m_Sigma(1)
  , m_Amplitude(1)
  ,m_lowLimit(0)
  ,m_highLimit(0)
{

}

GateDistributionTruncatedGaussian::GateDistributionTruncatedGaussian(const G4String& itsName,
                                                                     G4double mu,
                                                                     G4double sigma,
                                                                     G4double lowLimit,
                                                                     G4double highLimit)
:GateVDistribution(itsName), m_Mu(mu), m_Sigma(sigma), m_lowLimit(lowLimit), m_highLimit(highLimit) {
}
//___________________________________________________________________
GateDistributionTruncatedGaussian::~GateDistributionTruncatedGaussian()
{}
//___________________________________________________________________
G4double GateDistributionTruncatedGaussian::MinX() const
{
    return -DBL_MAX;
}
//___________________________________________________________________
G4double GateDistributionTruncatedGaussian::MinY() const
{
    return 0.;
}
//___________________________________________________________________
G4double GateDistributionTruncatedGaussian::MaxX() const
{
    return DBL_MAX;
}
//___________________________________________________________________
G4double GateDistributionTruncatedGaussian::MaxY() const
{
    return m_Amplitude*GateConstants::one_over_sqrt_2pi;
}
//___________________________________________________________________
G4double GateDistributionTruncatedGaussian::Value(G4double x) const
{
    return GateConstants::one_over_sqrt_2pi*exp(-(x-m_Mu)*(x-m_Mu)/(2.*m_Sigma*m_Sigma))*m_Amplitude;
}

//___________________________________________________________________ //Equivalent to ShootRandom
// Sample from a truncated Gaussian distribution
G4double GateDistributionTruncatedGaussian::shootTruncatedGaussian() const{

    // Adjust m_Sigma to preserve the final standard deviation
    G4double const adjustedSigma  = computeTruncatedSigma();



    G4double x;
    do {
        x = G4RandGauss::shoot(m_Mu,adjustedSigma);
    } while (x < m_lowLimit || x > m_highLimit);  // Reject samples outside [a, b]

    return x;
}
//___________________________________________________________________
void GateDistributionTruncatedGaussian::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent)
    	 <<"Mean : "         << m_Mu
         <<"  -- Sigma : "    << m_Sigma
         <<"  -- Amplitude : "<< m_Amplitude
	 << Gateendl;
}

//___________________________________________________________________
//___________________________________________________________________

//Static methods to be used without the need for a class object:
G4double GateDistributionTruncatedGaussian::shootRandom(G4double mu, G4double sigma, G4double lowLimit, G4double highLimit)
{   //return G4RandGauss::shoot(m_Mu,m_Sigma);

    // Adjust m_Sigma to preserve the final standard deviation
    G4double const adjustedSigma  = computeTruncatedSigmaStatic(mu, sigma, lowLimit, highLimit);



    G4double x;
    do {
        x = G4RandGauss::shoot(mu,adjustedSigma);
    } while (x < lowLimit || x > highLimit);  // Reject samples outside [a, b]

    return x;
}
//___________________________________________________________________
// Standard normal PDF
G4double GateDistributionTruncatedGaussian::pdf(G4double x){
    return exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
}
//___________________________________________________________________
// Standard normal CDF using the error function
G4double GateDistributionTruncatedGaussian::cdf(G4double x){
    return 0.5 * (1 + erf(x / sqrt(2.0)));
}
//___________________________________________________________________
// Compute the corrected standard deviation after truncation
G4double GateDistributionTruncatedGaussian::computeTruncatedSigmaStatic(G4double mu, G4double sigma, G4double lowLimit, G4double highLimit){
    double lowLim_std = (lowLimit - mu) / sigma;
    double hiLim_std = (highLimit - mu) / sigma;

    double phi_lowLim = pdf(lowLim_std);
    double phi_hiLim = pdf(hiLim_std);
    double Fl = cdf(lowLim_std);
    double Fh = cdf(hiLim_std);

    double Z = Fh - Fl;
    double mean_shift = (phi_lowLim - phi_hiLim) / Z;
    double variance_correction = 1 - (lowLim_std * phi_lowLim - hiLim_std * phi_hiLim) / Z - mean_shift * mean_shift;

    return sigma * sqrt(variance_correction);
}

//___________________________________________________________________
// Compute the corrected standard deviation after truncation
G4double GateDistributionTruncatedGaussian::computeTruncatedSigma() const{
    double lowLim_std = (m_lowLimit - m_Mu) / m_Sigma;
    double hiLim_std = (m_highLimit - m_Mu) / m_Sigma;

    double phi_lowLim = pdf(lowLim_std);
    double phi_hiLim = pdf(hiLim_std);
    double Fl = cdf(lowLim_std);
    double Fh = cdf(hiLim_std);

    double Z = Fh - Fl;
    double mean_shift = (phi_lowLim - phi_hiLim) / Z;
    double variance_correction = 1 - (lowLim_std * phi_lowLim - hiLim_std * phi_hiLim) / Z - mean_shift * mean_shift;

    return m_Sigma * sqrt(variance_correction);
}