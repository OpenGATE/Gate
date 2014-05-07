/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*
   \class  GateCreatorProcessFilterMessenger
   \author pierre.gueth@creatis.insa-lyon.fr
   */

#ifndef GATECREATORPROCESSFILTERMESSENGER_HH
#define GATECREATORPROCESSFILTERMESSENGER_HH

#include "globals.hh"

#include "G4UImessenger.hh"

#include "G4UIcmdWithAString.hh"

class GateCreatorProcessFilter;

class GateCreatorProcessFilterMessenger :
  public G4UImessenger
{
  public:
    GateCreatorProcessFilterMessenger(GateCreatorProcessFilter* filter);
    virtual ~GateCreatorProcessFilterMessenger();

    void BuildCommands(G4String base);
    void SetNewValue(G4UIcommand*, G4String);

  protected:
    GateCreatorProcessFilter *pFilter;

    G4UIcmdWithAString* pAddCreatorProcessCmd;
};

#endif
