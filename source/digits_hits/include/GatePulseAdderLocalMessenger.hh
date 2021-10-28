/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GatePulseAdderLocalMessenger_h
#define GatePulseAdderLocalMessenger_h 1

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

class GatePulseAdderLocal;

/*! \class  GatePulseAdderLocalMessenger
    \brief  Messenger for the GatePulseAdderLocal

    - GatePulseAdderLocalMessenger

    \sa GatePulseAdderLocal, GatePulseProcessorMessenger
*/
class GatePulseAdderLocalMessenger: public GatePulseProcessorMessenger
{
  public:
    GatePulseAdderLocalMessenger(GatePulseAdderLocal* itsPulseAdderLocal);
    ~GatePulseAdderLocalMessenger();

    void SetNewValue(G4UIcommand* aCommand, G4String aString);
    void SetNewValue2(G4UIcommand* aCommand, G4String aString);

    inline GatePulseAdderLocal* GetPulseAdderLocal()
      { return (GatePulseAdderLocal*) GetPulseProcessor(); }


private:

    G4UIcmdWithAString   *newVolCmd;

    G4int m_count;
    std::vector<G4String> m_name;
    std::vector<G4UIcmdWithAString*>         positionPolicyCmd;
    std::vector<G4UIdirectory*> m_volDirectory;

};

#endif
