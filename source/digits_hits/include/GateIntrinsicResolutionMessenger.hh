/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateIntrinsicResolution.cc for more detals
  */


/*! \class  GateIntrinsicResolutionMessenger
    \brief  Messenger for the GateIntrinsicResolution

    - GateIntrinsicResolution - by name.surname@email.com

    \sa GateIntrinsicResolution, GateIntrinsicResolutionMessenger
*/


#ifndef GateIntrinsicResolutionMessenger_h
#define GateIntrinsicResolutionMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateIntrinsicResolution;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithAString;

class GateIntrinsicResolutionMessenger : public GateClockDependentMessenger
{
public:
  
  GateIntrinsicResolutionMessenger(GateIntrinsicResolution*);
  ~GateIntrinsicResolutionMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateIntrinsicResolution* m_IntrinsicResolution;

  G4UIcmdWithADouble          *resolutionCmd;
  G4UIcmdWithADoubleAndUnit   *erefCmd;
  G4UIcmdWithADouble 		  *lightOutputCmd;
  G4UIcmdWithADouble          *coeffTECmd;
  G4UIcmdWithAString   *newFileQECmd;
  G4UIcmdWithADouble   *uniqueQECmd;

  G4UIcmdWithADouble   *varianceCmd;

  G4UIcmdWithADouble   *edgesFractionCmd;
  G4UIcmdWithADouble   *cornersFractionCmd;



};

#endif








