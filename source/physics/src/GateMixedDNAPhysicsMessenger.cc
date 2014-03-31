/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
  
  
#ifndef GateMixedDNAPhysicsMessenger_CC
#define GateMixedDNAPhysicsMessenger_CC

#include "GateMixedDNAPhysics.hh"

#include "GateMixedDNAPhysicsMessenger.hh"
#include "GateMiscFunctions.hh"

//----------------------------------------------------------------------------------------
GateMixedDNAPhysicsMessenger::GateMixedDNAPhysicsMessenger(GateMixedDNAPhysics * plMixed)
 :pMixedDNA(plMixed)
{}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
GateMixedDNAPhysicsMessenger::~GateMixedDNAPhysicsMessenger()
{   
  delete pSetDNAInRegion;
  delete pMixedDNA;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateMixedDNAPhysicsMessenger::BuildCommands(G4String base)
{
  G4String bb = base+"/SetDNAInRegion";
  pSetDNAInRegion = new G4UIcmdWithAString(bb,this);
  G4String guidance = "Select REGION with Geant4-DNA";
  pSetDNAInRegion->SetGuidance(guidance);
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
void GateMixedDNAPhysicsMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command == pSetDNAInRegion){
     pMixedDNA->defineRegionsWithDNA(param);
  }
}

#endif  /*end #define GateMixedDNAPhysicsMessenger_CC*/ 
