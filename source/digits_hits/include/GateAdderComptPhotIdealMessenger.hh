/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*!
  \class  GateAdderComptPhotIdealMessenger
  \brief  Messenger for the GateAdderComptPhotIdeal

  OK GND 2022

  Added to GND in November 2022 by olga.kochebina@cea.fr

  Last modification (Adaptation to GND): July 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#ifndef GateAdderComptPhotIdealMessenger_h
#define GateAdderComptPhotIdealMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"

class G4UIdirectory;
class GateAdderComptPhotIdeal;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class GateAdderComptPhotIdealMessenger : public GateClockDependentMessenger
{
public:
  
  GateAdderComptPhotIdealMessenger(GateAdderComptPhotIdeal*);
  virtual ~GateAdderComptPhotIdealMessenger();
  
  inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

  
private:
  GateAdderComptPhotIdeal* m_GateAdderComptPhotIdeal;
  G4UIcmdWithABool* pRejectionPolicyCmd;


};

#endif








