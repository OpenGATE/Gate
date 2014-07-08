/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateProductionActorMessenger
  \author pierre.gueth@creatis.insa-lyon.fr
*/

#ifndef GATEPRODUCTIONACTORMESSENGER_HH
#define GATEPRODUCTIONACTORMESSENGER_HH

#include "GateImageActorMessenger.hh"

class GateProductionActor;

//-----------------------------------------------------------------------------
class GateProductionActorMessenger : public GateImageActorMessenger
{
  public:

    //-----------------------------------------------------------------------------
    /// Constructor with pointer on the associated sensor
    GateProductionActorMessenger(GateProductionActor * v);

    //-----------------------------------------------------------------------------
    /// Destructor
    virtual ~GateProductionActorMessenger();

}; // end class GateProductionActorMessenger
//-----------------------------------------------------------------------------

#endif /* end #define GATEPRODUCTIONACTORMESSENGER_HH */
