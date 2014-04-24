/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSignalHandler_h
#define GateSignalHandler_h 1

#include "globals.hh"

/*! \namespace  GateSignalHandler
    \brief  Namespace to catch and process system signals

    - GateSignalHandler - by Daniel.Strul@iphe.unil.ch (June 7 2002)

    - The GateSignalHandler namespace is a collection of routines allowing to catch and process system signals.

    - For now, its only task is to catch SCTRL-\ (SIGQUIT) so that the user can stop an acquisition
      and get back in idle mode.

*/
namespace GateSignalHandler
{
     //! Install the signal handlers
    G4int Install(void);

    //! Handles the signal SIGQUIT (CTRL-\).
    //! When a BeamOn/StartDAQ is running, aborts the current run and stops the DAQ, returning GATE in Idle state.
    //! In the other states, the signal is ignored.
    void QuitSignalHandler(int sig);
    void PrintSimulationStatus(int sig);

    //! Ignore signal
    void IgnoreSignalHandler(int sig);
}

#endif
