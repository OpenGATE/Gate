/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \class  GateEnergyFilterMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEENEFILTERMESSENGER_HH
#define GATEENEFILTERMESSENGER_HH

#include "globals.hh"

#include "G4UImessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"


class GateEnergyFilter;

class GateEnergyFilterMessenger : public  G4UImessenger
{
public:
  GateEnergyFilterMessenger(GateEnergyFilter* eneFilter);
  virtual ~GateEnergyFilterMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateEnergyFilter * pEnergyFilter;

  G4UIcmdWithADoubleAndUnit * pSetEminCmd;
  G4UIcmdWithADoubleAndUnit * pSetEmaxCmd;
  G4UIcmdWithoutParameter* pInvertCmd;
};

#endif /* end #define GATEENEFILTERMESSENGER_HH */
