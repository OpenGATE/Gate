/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateVDistribution.hh"
#include "GateDistributionListManager.hh"

GateVDistribution::GateVDistribution(const G4String& itsName)
  : GateNamedObject(itsName)
{
    GateDistributionListManager::GetInstance()->RegisterDistribution(this);
}

GateVDistribution::~GateVDistribution()
{
}

G4double GateVDistribution::Value2D(G4double x, G4double y)const {


    G4cerr << "Warning: GateVDistribution::Value2D always returns 0 for Gaussian, flat (uniform), and exponential distributions." <<G4endl;

    return 0.0;
}
