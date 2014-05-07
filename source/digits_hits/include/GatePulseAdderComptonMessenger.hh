/*----------------------
   OpenGATE Collaboration

   jbmichaud@videotron.ca

   Copyright (C) 2009 Universite de Sherbrooke

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*!

  \file GatePulseAdderComptonMessenger.hh

  $Log: GatePulseAdderMessenger.hh,v $

  Revision 1.1  2009/04/16  jbmichaud
  New class for summing pulses in an exact Compton kinetics context.

  \brief Class GatePulseAdderComptonMessenger
  \brief By jbmichaud@videotron.ca
  \brief $Id: GatePulseAdderMessenger.hh,v 1.1 2009/04/17 15:33:24 jbmichaud Exp $
*/

#ifndef GatePulseAdderComptonMessenger_h
#define GatePulseAdderComptonMessenger_h 1

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

class GatePulseAdderCompton;

/*! \class  GatePulseAdderComptonMessenger
    \brief  Messenger for the GatePulseAdderCompton

    - GatePulseAdderComptonMessenger - by jbmichaud@videotron.ca

    \sa GatePulseAdderCompton, GatePulseProcessorMessenger
*/
class GatePulseAdderComptonMessenger: public GatePulseProcessorMessenger
{
  public:
    GatePulseAdderComptonMessenger(GatePulseAdderCompton* itsPulseAdder);
    inline ~GatePulseAdderComptonMessenger() {}

    void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GatePulseAdderCompton* GetPulseAdderCompton()
      { return (GatePulseAdderCompton*) GetPulseProcessor(); }

};

#endif
