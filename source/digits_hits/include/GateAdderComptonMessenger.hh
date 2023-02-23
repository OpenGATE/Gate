/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*! \class  GateAdderComptonMessenger
    \brief  Messenger for the GateAdderCompton

    - GateAdderCompton - by jbmichaud@videotron.ca

	$Log: GateAdderMessenger.hh,v $

  	Revision 1.1  2009/04/16  jbmichaud
    New class for summing digis in an exact Compton kinetics context.

    \brief Class GateAdderComptonMessenger
    \brief By jbmichaud@videotron.ca
    \brief $Id: GateAdderMessenger.hh,v 1.1 2009/04/17 15:33:24 jbmichaud Exp $

  	OK: added to GND in Jan2023

    \sa GateAdderCompton, GateAdderComptonMessenger
*/


#ifndef GateAdderComptonMessenger_h
#define GateAdderComptonMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateAdderCompton;
class G4UIcmdWithAString;

class GateAdderComptonMessenger : public GateClockDependentMessenger
{
public:
  
  GateAdderComptonMessenger(GateAdderCompton*);
  ~GateAdderComptonMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateAdderCompton* m_AdderCompton;


};

#endif








