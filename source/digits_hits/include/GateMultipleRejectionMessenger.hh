/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*!
  \class  GateMultipleRejectionMessenger
  \brief  Messenger for the GateMultipleRejection

  Last modification (Adaptation to GND): August 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#ifndef GateMultipleRejectionMessenger_h
#define GateMultipleRejectionMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateMultipleRejection;
class G4UIcmdWithAString;
class G4UIcmdWithABool;

class GateMultipleRejectionMessenger : public GateClockDependentMessenger
{
public:
  
  GateMultipleRejectionMessenger(GateMultipleRejection*);
  ~GateMultipleRejectionMessenger();
  

  inline void SetNewValue(G4UIcommand* aCommand, G4String aString);
  inline void SetNewValue2(G4UIcommand* aCommand, G4String aString);


  
private:
  GateMultipleRejection* m_MultipleRejection;
  G4UIcmdWithABool   *newEventRejCmd;
  G4UIcmdWithAString *newMultiDefCmd;
  std::vector<G4UIdirectory*> m_volDirectory;
  std::vector<G4UIcmdWithABool *>   MultipleRejectionPolicyCmd;
  std::vector<G4UIcmdWithAString *>   MultipleDefinitionCmd;
  std::vector<G4String> m_name;
  G4int m_count;


};

#endif








