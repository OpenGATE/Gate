/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateDistributionExponential.hh"

#include "GateDistributionExponentialMessenger.hh"
#include <math.h>
#include <CLHEP/Random/Randomize.h>
#include <CLHEP/Random/RandExponential.h>
#include "Randomize.hh"
#include "GateTools.hh"


GateDistributionExponential::GateDistributionExponential(const G4String& itsName)
  : GateVDistribution(itsName)
  , m_Lambda(1)
  , m_Amplitude(1)
{
    m_messenger = new GateDistributionExponentialMessenger(this,itsName);
}
//___________________________________________________________________
GateDistributionExponential::~GateDistributionExponential()
{
    delete m_messenger;
}
//___________________________________________________________________
G4double GateDistributionExponential::MinX() const
{
    return -DBL_MAX;
}
//___________________________________________________________________
G4double GateDistributionExponential::MinY() const
{
    return 0.;
}
//___________________________________________________________________
G4double GateDistributionExponential::MaxX() const
{
    return DBL_MAX;
}
//___________________________________________________________________
G4double GateDistributionExponential::MaxY() const
{
    return m_Amplitude/m_Lambda;
}
//___________________________________________________________________
G4double GateDistributionExponential::Value(G4double x) const
{
    return (x<0) ? 0 : exp(-x/m_Lambda)*m_Amplitude/m_Lambda;
}
//___________________________________________________________________
G4double GateDistributionExponential::ShootRandom() const
{
    return (m_Lambda>0) ? CLHEP::RandExponential::shoot(m_Lambda) : 0;
}
//___________________________________________________________________
void GateDistributionExponential::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent)
    	 <<"Lambda : "         << m_Lambda
         <<"  -- Amplitude : "<< m_Amplitude
	 <<G4endl;
}
