/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  \class  GateTimeDelay

  Digitizer module for simulating a TimeDelay
  The user can choose a specific TimeDelay for each tracked volume.
  \sa GateTimeDelay, GateTimeDelayMessenger
    
  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#ifndef GateTimeDelay_h
#define GateTimeDelay_h 1

#include <iostream>
#include <vector>

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateTimeDelayMessenger.hh"
#include "GateSinglesDigitizer.hh"

#include "GateMaps.hh"

class GateTimeDelay : public GateVDigitizerModule
{
public:
  
  GateTimeDelay(GateSinglesDigitizer *digitizer, G4String name);
  ~GateTimeDelay();
  
  void Digitize() override;

  
  void SetTimeDelay(G4double val)   { m_TimeDelay = val; };


  void DescribeMyself(size_t );

protected:

  G4double 	 m_TimeDelay;

private:
  GateDigi* m_outputDigi;

  GateTimeDelayMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;


};

#endif








