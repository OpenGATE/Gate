/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \class  GateIDFilterMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEIDFILTERMESSENGER_HH
#define GATEIDFILTERMESSENGER_HH

#include "globals.hh"

#include "G4UImessenger.hh"

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithoutParameter.hh"


class GateIDFilter;

class GateIDFilterMessenger : public  G4UImessenger
{
public:
  GateIDFilterMessenger(GateIDFilter* idFilter);
  virtual ~GateIDFilterMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateIDFilter * pIDFilter;

  G4UIcmdWithAnInteger* pAddIDCmd;
  G4UIcmdWithAnInteger* pAddParentIDCmd;
  G4UIcmdWithoutParameter* pInvertCmd;
};

#endif /* end #define GATEIDFILTERMESSENGER_HH */
