/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \class  GateVolumeFilterMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEVOLUMEFILTERMESSENGER_HH
#define GATEVOLUMEFILTERMESSENGER_HH

#include "globals.hh"

#include "G4UImessenger.hh"

#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithoutParameter.hh"


class GateVolumeFilter;

class GateVolumeFilterMessenger : public  G4UImessenger
{
public:
  GateVolumeFilterMessenger(GateVolumeFilter* idFilter);
  virtual ~GateVolumeFilterMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateVolumeFilter * pVolumeFilter;

  G4UIcmdWithAString* pAddVolumeCmd;
  G4UIcmdWithoutParameter* pInvertCmd;
};

#endif /* end #define GATEVOLUMEFILTERMESSENGER_HH */
