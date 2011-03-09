/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \brief Class GateProductionActor : 
  \brief compute production count for filtered particles
 */

#ifndef GATEPRODUCTIONACTOR_CC
#define GATEPRODUCTIONACTOR_CC

#include "GateProductionActor.hh"

#include "GateMiscFunctions.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateProductionActor::GateProductionActor(G4String name, G4int depth) :
  GateVImageActor(name,depth), pMessenger(NULL)
{
  GateDebugMessageInc("Actor",4,"GateProductionActor() -- begin"<<G4endl);

  pMessenger = new GateProductionActorMessenger(this);

  GateDebugMessageDec("Actor",4,"GateProductionActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor 
GateProductionActor::~GateProductionActor() 
{
  GateDebugMessageInc("Actor",4,"~GateProductionActor() -- begin"<<G4endl);
 
  GateDebugMessageDec("Actor",4,"~GateProductionActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateProductionActor::Construct()
{
  GateVImageActor::Construct();

  // Enable callbacks
  EnablePreUserTrackingAction(true);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Start of track callback
void GateProductionActor::UserPreTrackActionInVoxel(const int index, const G4Track* /*track*/)
{
  mImage.GetValue(index)++;
}


#endif /* end #define GATEPRODUCTIONACTOR_CC */
