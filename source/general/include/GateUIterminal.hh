/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*----------------------

  GATE - Geant4 Application for Tomographic Emission
  OpenGATE Collaboration

  Daniel Strul <daniel.strul@iphe.unil.ch>

  Copyright (C) 2002 UNIL/IPHE, CH-1015 Lausanne

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateUIterminal_h
#define GateUIterminal_h 1

#include "GateConfiguration.h"
#include "G4UIterminal.hh"

class GateUIterminal : public G4UIterminal
{

public:
  GateUIterminal(G4VUIshell* aShell=0)
    : G4UIterminal(aShell)
  {}

  ~GateUIterminal() {}

  virtual G4int ReceiveG4cout( const G4String& coutString);
  virtual G4int ReceiveG4cerr( const G4String& cerrString);

};

#endif
