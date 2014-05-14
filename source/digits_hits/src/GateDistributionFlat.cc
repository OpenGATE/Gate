/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateDistributionFlat.hh"

#include "GateDistributionFlatMessenger.hh"
#include <math.h>
//#include <CLHEP/Random/RandFlat.h>
#include "Randomize.hh"
#include "GateTools.hh"


GateDistributionFlat::GateDistributionFlat(const G4String& itsName)
  : GateVDistribution(itsName)
  , m_Min(0)
  , m_Max(1)
  , m_Amplitude(1)
{
    m_messenger = new GateDistributionFlatMessenger(this,itsName);
}
//___________________________________________________________________
GateDistributionFlat::~GateDistributionFlat()
{
    delete m_messenger;
}
//___________________________________________________________________
G4double GateDistributionFlat::MinX() const
{
    return m_Min;
}
//___________________________________________________________________
G4double GateDistributionFlat::MinY() const
{
    return 0;
}
//___________________________________________________________________
G4double GateDistributionFlat::MaxX() const
{
    return m_Max;
}
//___________________________________________________________________
G4double GateDistributionFlat::MaxY() const
{
    return GetAmplitude()/(m_Max-m_Min);
}
//___________________________________________________________________
G4double GateDistributionFlat::Value(G4double x) const
{
    return (x>m_Min && x<m_Max) ? MaxY() : 0;
}
//___________________________________________________________________
G4double GateDistributionFlat::ShootRandom() const
{
    return (m_Min + (m_Max - m_Min)*G4UniformRand());
}
//___________________________________________________________________
void GateDistributionFlat::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent)
    	 <<"Min : "         << m_Min
         <<"  -- Max : "    << m_Max
         <<"  -- Amplitude : "<< m_Amplitude
	 <<G4endl;
}
