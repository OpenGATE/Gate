/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateDiscretizerMessenger_h
#define GateDiscretizerMessenger_h 1

#include "GatePulseProcessorMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateDiscretizer;

/*! \class  GateDiscretizerMessenger
    \brief  Messenger for the GateDiscretizer

    - GateDiscretizerMessenger - by dguez@cea.fr

    \sa GateDiscretizer, GatePulseProcessorMessenger
*/
class GateDiscretizerMessenger: public GatePulseProcessorMessenger
{
  public:
    GateDiscretizerMessenger(GateDiscretizer* itsDiscretizer);
    virtual ~GateDiscretizerMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateDiscretizer* GetDiscretizer()
      { return (GateDiscretizer*) GetPulseProcessor(); }

  private:
};

#endif
