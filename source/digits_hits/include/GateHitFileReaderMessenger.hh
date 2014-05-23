/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateHitFileReaderMessenger_h
#define GateHitFileReaderMessenger_h 1

#include "GateConfiguration.h"

//e #ifdef G4ANALYSIS_USE_ROOT

#include "GateClockDependentMessenger.hh"

class GateHitFileReader;


/*! \class GateHitFileReaderMessenger
    \brief Messenger used to command a GateHitFileReader

    - GateHitFileReaderMessenger - by Daniel.Strul@iphe.unil.ch

    - The GateHitFileReaderMessenger inherits from the abilities/responsabilities
      of the GateClockDependentMessenger base-class, i.e. the creation and management
      of a Gate UI directory for a Gate object, plus the UI command 'describe'

    - In addition, it proposes and manages commands specific to the hit-file reader:
      definition of the name of the hit file

*/
class GateHitFileReaderMessenger: public GateClockDependentMessenger
{
  public:
    GateHitFileReaderMessenger(GateHitFileReader* itsHitFileReader);
   ~GateHitFileReaderMessenger();

    void SetNewValue(G4UIcommand*, G4String);

    //! Get the clock-dependent object
    inline GateHitFileReader* GetHitFileReader()
      { return (GateHitFileReader*) GetClockDependent(); }

  protected:
    G4UIcmdWithAString*      SetFileNameCmd;
};

//e #endif
#endif
