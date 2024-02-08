/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  \class  GateMaterialFilterMessenger
*/

#ifndef GATEMATFILTERMESSENGER_HH
#define GATEMATFILTERMESSENGER_HH

#include "globals.hh"
#include "G4UImessenger.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithoutParameter.hh"


class GateMaterialFilter;

class GateMaterialFilterMessenger : public  G4UImessenger
{
public:
  GateMaterialFilterMessenger(GateMaterialFilter* matFilter);
  virtual ~GateMaterialFilterMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateMaterialFilter * pMaterialFilter;

  G4UIcmdWithAString* pAddMaterialCmd;
  G4UIcmdWithoutParameter* pInvertCmd;
};

#endif /* end #define GATEMATFILTERMESSENGER_HH */
