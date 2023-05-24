/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*! \class  GateDoIModelsMessenger
    \brief  Messenger for the GateDoIModels

    \sa GateDoIModels, GateDoIModelsMessenger

    Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com

*/


#ifndef GateDoIModelsMessenger_h
#define GateDoIModelsMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
#include "GateVDoILaw.hh"

class GateDoIModels;
class G4UIcmdWithAString;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;
class G4UIdirectory;
class G4UIcmdWithoutParameter;

class GateDoIModelsMessenger : public GateClockDependentMessenger
{
public:

  GateDoIModelsMessenger(GateDoIModels*);
  ~GateDoIModelsMessenger();

  void SetNewValue(G4UIcommand*, G4String);

private:
  G4UIcmdWith3Vector *axisCmd;
  GateVDoILaw* CreateDoILaw(const G4String& law);
  G4UIcmdWithAString *lawCmd;
  GateDoIModels* m_DoIModels;
  G4UIcmdWithAString          *DoICmd;


};

#endif









