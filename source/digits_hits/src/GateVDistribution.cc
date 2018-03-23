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
