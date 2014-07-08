/*----------------------
   OpenGATE Collaboration

   jbmichaud@videotron.ca

   Copyright (C) 2009 Universite de Sherbrooke

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*!

  \file GatePulseAdderComptonMessenger.cc

  $Log: GatePulseAdderMessenger.hh,v $

  Revision 1.1  2009/04/16  jbmichaud
  New class for summing pulses in an exact Compton kinetics context.

  \brief Class GatePulseAdderComptonMessenger
  \brief By jbmichaud@videotron.ca
  \brief $Id: GatePulseAdderMessenger.hh,v 1.1 2009/04/17 15:33:24 jbmichaud Exp $
*/


#include "GatePulseAdderComptonMessenger.hh"

#include "GatePulseAdderCompton.hh"

GatePulseAdderComptonMessenger::GatePulseAdderComptonMessenger(GatePulseAdderCompton* itsPulseAdder)
    : GatePulseProcessorMessenger(itsPulseAdder)
{
}


void GatePulseAdderComptonMessenger::SetNewValue(G4UIcommand* aCommand, G4String aString)
{
  GatePulseProcessorMessenger::SetNewValue(aCommand,aString);
}
