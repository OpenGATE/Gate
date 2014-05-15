/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#ifndef GateFastAnalysisMessenger_h
#define GateFastAnalysisMessenger_h 1

#include "GateOutputModuleMessenger.hh"

class GateFastAnalysis;

//! Messenger belonging to GateFastAnalysis
/**
  * Introduces no new macro commands, only commands of GateOutputModuleMessenger
  * are inherited.
  * */
class GateFastAnalysisMessenger: public GateOutputModuleMessenger
{
  public:
    GateFastAnalysisMessenger(GateFastAnalysis* gateFastAnalysis);
   ~GateFastAnalysisMessenger();

    virtual void SetNewValue(G4UIcommand*, G4String);

  protected:
    GateFastAnalysis* m_gateFastAnalysis;

};

#endif

#endif
