/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GateSystemFilterMessenger_h
#define GateSystemFilterMessenger_h 1

#include "GatePulseProcessorMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class GateSystemFilter;

/*! \class GateSystemFilterMessenger
    \brief The messenger of GateSystemFilter class (multi-system approach)

    - GateSystemFilterMessenger -by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr
 */

class GateSystemFilterMessenger: public GatePulseProcessorMessenger
{

  public:
     GateSystemFilterMessenger(GateSystemFilter* itsSystemFilter);
     ~GateSystemFilterMessenger();

    void SetNewValue(G4UIcommand* aCommand, G4String aString);
    void ObtainCandidates();

    inline GateSystemFilter* GetSystemFilter()
      { return (GateSystemFilter*) GetPulseProcessor(); }

   private:

      G4UIcmdWithAString* m_SetSystemNameCmd;
      G4String m_insertedSystems;
};

#endif
