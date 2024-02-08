/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \class  GateAngleFilterMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GATEANGLEFILTERMESSENGER_HH
#define GATEANGLEFILTERMESSENGER_HH

#include "globals.hh"

#include "G4UImessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWithoutParameter.hh"


class GateAngleFilter;

class GateAngleFilterMessenger : public  G4UImessenger
{
public:
  GateAngleFilterMessenger(GateAngleFilter* angleFilter);
  virtual ~GateAngleFilterMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateAngleFilter * pAngleFilter;

  G4UIcmdWithADoubleAndUnit * pSetAngleCmd;
  G4UIcmdWith3Vector        * pSetDirectionCmd;
  G4UIcmdWithoutParameter* pInvertCmd;
};

#endif /* end #define GATEANGLEFILTERMESSENGER_HH */
