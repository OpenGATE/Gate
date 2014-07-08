/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GATEPRODUCTIONACTORMESSENGER_CC
#define GATEPRODUCTIONACTORMESSENGER_CC

#include "GateProductionActorMessenger.hh"
#include "GateProductionActor.hh"

//-----------------------------------------------------------------------------
GateProductionActorMessenger::GateProductionActorMessenger(GateProductionActor * v)
: GateImageActorMessenger(v)
{
  // build specific command here
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateProductionActorMessenger::~GateProductionActorMessenger()
{
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEPRODUCTIONACTORMESSENGER_CC */
