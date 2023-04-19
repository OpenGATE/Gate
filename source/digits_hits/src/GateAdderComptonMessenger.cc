/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*! \class  GateAdderComptonMessenger
    \brief  Messenger for the GateAdderCompton

    - GateAdderCompton - by Daniel.Strul@iphe.unil.ch, jbmichaud@videotron.ca
    OK: added to GND in Jan2023

    \sa GateAdderCompton, GateAdderComptonMessenger
*/

#include "GateAdderComptonMessenger.hh"
#include "GateAdderCompton.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"



GateAdderComptonMessenger::GateAdderComptonMessenger (GateAdderCompton* AdderCompton)
:GateClockDependentMessenger(AdderCompton),
 	 m_AdderCompton(AdderCompton)
{
}


GateAdderComptonMessenger::~GateAdderComptonMessenger()
{
}


void GateAdderComptonMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{

	GateClockDependentMessenger::SetNewValue(aCommand,newValue);

}













