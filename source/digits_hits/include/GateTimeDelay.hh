/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

/*!
  \class  GateTimeDelay

  Digitizer module for simulating a DoI model
  The user can choose the axes for each tracked volume.
  
  \sa GateTimeDelay, GateTimeDelayMessenger
  OK GND 2022

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


  // *******implement your methods here
  
  void SetTimeDelay(G4double val)   { m_TimeDelay = val; };


  void DescribeMyself(size_t );

protected:
  // *******implement your parameters here

  G4double 	 m_TimeDelay;

private:
  GateDigi* m_outputDigi;

  GateTimeDelayMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;


};

#endif








