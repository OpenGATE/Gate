/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateDistributionGauss.hh"

#include "GateDistributionGaussMessenger.hh"
#include <math.h>
//#include <CLHEP/Random/RandGauss.h>
#include "Randomize.hh"
#include "GateTools.hh"


GateDistributionGauss::GateDistributionGauss(const G4String& itsName)
  : GateVDistribution(itsName)
  , m_Mean(0)
  , m_Sigma(1)
  , m_Amplitude(1)
{
    m_messenger = new GateDistributionGaussMessenger(this,itsName);
}
//___________________________________________________________________
GateDistributionGauss::~GateDistributionGauss()
{
    delete m_messenger;
}
//___________________________________________________________________
G4double GateDistributionGauss::MinX() const
{
    return -DBL_MAX;
}
//___________________________________________________________________
G4double GateDistributionGauss::MinY() const
{
    return 0.;
}
//___________________________________________________________________
G4double GateDistributionGauss::MaxX() const
{
    return DBL_MAX;
}
//___________________________________________________________________
G4double GateDistributionGauss::MaxY() const
{
    static const G4double one_over_sqrt_2pi = 0.39894228040143267794;
    return m_Amplitude*one_over_sqrt_2pi;
}
//___________________________________________________________________
G4double GateDistributionGauss::Value(G4double x) const
{
    static const G4double one_over_sqrt_2pi = 0.39894228040143267794;
    return one_over_sqrt_2pi*exp(-(x-m_Mean)*(x-m_Mean)/(2.*m_Sigma*m_Sigma))*m_Amplitude;
}
//___________________________________________________________________
G4double GateDistributionGauss::ShootRandom() const
{
    return G4RandGauss::shoot(m_Mean,m_Sigma);
}
//___________________________________________________________________
void GateDistributionGauss::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent)
    	 <<"Mean : "         << m_Mean
         <<"  -- Sigma : "    << m_Sigma
         <<"  -- Amplitude : "<< m_Amplitude
	 <<G4endl;
}
