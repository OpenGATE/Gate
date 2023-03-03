/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*! \class  GateDigitizerMergerMessenger
    \brief  Messenger for the GateDigitizerMerger

    - GateDigitizerMerger - by olga.kochebina@cea.fr 03/03/23

    \sa GateDigitizerMerger, GateDigitizerMergerMessenger
*/


#ifndef GateDigitizerMergerMessenger_h
#define GateDigitizerMergerMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateDigitizerMerger;
class G4UIcmdWithAString;

class GateDigitizerMergerMessenger : public GateClockDependentMessenger
{
public:
  
  GateDigitizerMergerMessenger(GateDigitizerMerger*);
  ~GateDigitizerMergerMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateDigitizerMerger* m_DigitizerMerger;
  G4UIcmdWithAString          *addCollCmd;


};

#endif








