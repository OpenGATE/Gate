/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateHybridMultiplicityActorMessenger
  \author francois.smekens@creatis.insa-lyon.fr
*/

#ifndef GATEHYBRIDMULTIPLICITYACTORMESSENGER_HH
#define GATEHYBRIDMULTIPLICITYACTORMESSENGER_HH

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithAString.hh"
#include "GateImageActorMessenger.hh"

class GateHybridMultiplicityActor;
class GateHybridMultiplicityActorMessenger : public GateActorMessenger
{
public:
  
  GateHybridMultiplicityActorMessenger(GateHybridMultiplicityActor* sensor);
  virtual ~GateHybridMultiplicityActorMessenger();
  
  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  
  GateHybridMultiplicityActor * pMultiplicityActor;
  
  G4UIcmdWithAnInteger * pSetPrimaryMultiplicityCmd;
  G4UIcmdWithAnInteger * pSetSecondaryMultiplicityCmd;
};

#endif /* end #define GATEHYBRIDMULTIPLICITYACTORMESSENGER_HH*/
