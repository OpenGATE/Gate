/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSPSAngDistribution.hh"


GateSPSAngDistribution::GateSPSAngDistribution()
{

}

GateSPSAngDistribution::~GateSPSAngDistribution()
{

}

void GateSPSAngDistribution::SetFocusPointCopy(G4ThreeVector p)
{
  FocusPointCopy = p;
}

G4ThreeVector GateSPSAngDistribution::GetFocusPointCopy()
{
  return FocusPointCopy;
}
