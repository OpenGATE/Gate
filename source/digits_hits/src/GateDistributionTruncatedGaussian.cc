/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateDistributionTruncatedGaussian.hh"

#include "GateDistributionTruncatedGaussianMessenger.hh"
#include <math.h>
//#include <CLHEP/Random/RandGauss.h>
#include "Randomize.hh"
#include "GateTools.hh"
#include "GateConstants.hh"


GateDistributionTruncatedGaussian::GateDistributionTruncatedGaussian(const G4String& itsName)
  : GateVDistribution(itsName)
  , m_Mean(0)
  , m_Sigma(1)
  , m_Amplitude(1)
{

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
    return GateConstants::one_over_sqrt_2pi*exp(-(x-m_Mean)*(x-m_Mean)/(2.*m_Sigma*m_Sigma))*m_Amplitude;
}
//___________________________________________________________________
G4double GateDistributionTruncatedGaussian::ShootRandom() const
{
    //return G4RandGauss::shoot(m_Mean,m_Sigma);
	return GateDistributionTruncatedGaussian::truncatedGaussian();
}

//___________________________________________________________________
void GateDistributionTruncatedGaussian::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent)
    	 <<"Mean : "         << m_Mean
         <<"  -- Sigma : "    << m_Sigma
         <<"  -- Amplitude : "<< m_Amplitude
	 << Gateendl;
}


// Standard normal PDF
double GateDistributionTruncatedGaussian::pdf(double x) {
    return exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
}

// Standard normal CDF using the error function
double GateDistributionTruncatedGaussian::cdf(double x) {
    return 0.5 * (1 + erf(x / sqrt(2.0)));
}

// Compute the corrected standard deviation after truncation
double GateDistributionTruncatedGaussian::computeTruncatedSigma() {
    double lowLim_std = (lowLimit - m_mu) / m_Sigma;
    double hiLim_std = (highLimit - m_mu) / m_Sigma;

    double phi_lowLim = pdf(lowLim_std);
    double phi_hiLim = pdf(hiLim_std);
    double Fl = cdf(lowLim_std);
    double Fh = cdf(hiLim_std);

    double Z = Fh - Fl;
    double mean_shift = (phi_lowLim - phi_hiLim) / Z;
    double variance_correction = 1 - (lowLim_std * phi_lowLim - hiLim_std * phi_hiLim) / Z - mean_shift * mean_shift;

    return m_Sigma * sqrt(variance_correction);
}

// Sample from a truncated Gaussian distribution
double GateDistributionTruncatedGaussian::truncatedGaussian() {

    // Adjust m_Sigma to preserve the final standard deviation
    double adjustedSigma = computeTruncatedSigma(m_mu, m_Sigma, lowLimit, highLimit);



    double x;
    do {
        x = G4RandGauss::shoot(m_Mean,adjustedSigma);
    } while (x < lowLimit || x > highLimit);  // Reject samples outside [a, b]

    caca
    return x;
}


/*
 *
 *
//___________________________________________________________________

// Standard normal CDF using the error function
double cdf(double x) {
    return 0.5 * (1 + erf(x / sqrt(2.0)));
}

// Standard normal inverse CDF approximation (Probit function)
double inverseCDF(double p) {
    return sqrt(2.0) * std::erfinv(2.0 * p - 1.0);
}

// Sample from a truncated Gaussian without rejection sampling
double truncatedGaussian(double m_mu, double m_Sigma, double lowLimit, double highLimit) {

    double lowLim_std = (lowLim - m_mu) / m_Sigma;
    double hiLim_std = (hiLim - m_mu) / m_Sigma;

    double F_low = cdf(lowLim_std);
    double F_high = cdf(hiLim_std);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> uniform(F_low, F_high);

    // Sample in [F(a), F(b)] and apply inverse CDF
    double u = uniform(gen);
    double z = inverseCDF(u); // Inverse CDF to get standard normal sample

    return m_mu + m_Sigma * z;
}
*
*/

