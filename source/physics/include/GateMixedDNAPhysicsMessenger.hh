/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
  
  
  
#ifndef GateMixedDNAPhysicsMessenger_HH
#define GateMixedDNAPhysicsMessenger_HH

#include "globals.hh"

#include "G4UImessenger.hh"
#include "GateUIcmdWith2String.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "GateUIcmdWithAStringAndAnInteger.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"


class GateMixedDNAPhysics;

class GateMixedDNAPhysicsMessenger:public G4UImessenger
{
public:
  GateMixedDNAPhysicsMessenger(GateMixedDNAPhysics* plMixed);
  ~GateMixedDNAPhysicsMessenger();
  
  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:

 GateMixedDNAPhysics* pMixedDNA;
 G4UIcmdWithAString * pSetDNAInRegion;
}; 

#endif  /*end #define GateMixedDNAPhysicsMessenger_HH*/ 


