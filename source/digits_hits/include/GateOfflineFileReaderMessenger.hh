/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateOfflineFileReaderMessenger_h
#define GateOfflineFileReaderMessenger_h 1

#include "GateConfiguration.h"

//e #ifdef G4ANALYSIS_USE_ROOT

#include "GateClockDependentMessenger.hh"

class GateOfflineFileReader;


/*! \class GateOfflineFileReaderMessenger
    \brief Messenger used to command a GateOfflineFileReader

    - GateOfflineFileReaderMessenger - by Daniel.Strul@iphe.unil.ch

    - The GateOfflineFileReaderMessenger inherits from the abilities/responsabilities
      of the GateClockDependentMessenger base-class, i.e. the creation and management
      of a Gate UI directory for a Gate object, plus the UI command 'describe'

    - In addition, it proposes and manages commands specific to the hit-file reader:
      definition of the name of the hit file

*/
class GateOfflineFileReaderMessenger: public GateClockDependentMessenger
{
  public:
    GateOfflineFileReaderMessenger(GateOfflineFileReader* itsOfflineFileReader);
   ~GateOfflineFileReaderMessenger();

    void SetNewValue(G4UIcommand*, G4String);

    //! Get the clock-dependent object
    inline GateOfflineFileReader* GetOfflineFileReader()
      { return (GateOfflineFileReader*) GetClockDependent(); }

  protected:
    G4UIcmdWithAString*      SetFileNameCmd;
};

//e #endif
#endif
