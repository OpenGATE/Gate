/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \class  GateEmCalculatorActorMessenger
  \author loic.grevillot@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEEmCalculatorActorMESSENGER_HH
#define GATEEmCalculatorActorMESSENGER_HH

#include "globals.hh"
#include "GateActorMessenger.hh"

class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithAString;
class G4UIcommand;

class GateEmCalculatorActor;
class GateEmCalculatorActorMessenger : public GateActorMessenger
{
public:
  GateEmCalculatorActorMessenger(GateEmCalculatorActor* sensor);
  virtual ~GateEmCalculatorActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateEmCalculatorActor * pEmCalculatorActor;

  G4UIcmdWithADoubleAndUnit * pSetEnergyCmd;
  G4UIcmdWithAString * pSetParticleNameCmd;
  G4UIcommand * pIonCmd;
};

#endif /* end #define GATEEmCalculatorActorMESSENGER_HH*/
