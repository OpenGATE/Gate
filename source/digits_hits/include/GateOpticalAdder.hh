/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

/*! \class  GateOpticalAdder
    \brief  Pulse-processor for adding/grouping pulses per volume, only digits caused by optical photons are added.

    - GateOpticalAdder - by d.j.vanderlaan@iri.tudelft.nl

    - For each volume where there was one or more input pulse, we get exactly
      one output pulse, whose energy is the total number of input-digis (not the total energy like in the
      GateAdder, but the total number!), and whose position is the centroid of the input-digis positions.

	- Added to GND in Feb. 2023 by OK
*/
#include "GateConfiguration.h"
#ifdef GATE_USE_OPTICAL

#ifndef GateOpticalAdder_h
#define GateOpticalAdder_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateOpticalAdderMessenger.hh"
#include "GateSinglesDigitizer.hh"


class GateOpticalAdder : public GateVDigitizerModule
{
public:
  
  GateOpticalAdder(GateSinglesDigitizer *digitizer, G4String name);
  ~GateOpticalAdder();
  
  void Digitize() override;

  void DescribeMyself(size_t );


private:
  GateDigi* m_outputDigi;

  GateOpticalAdderMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;


};

#endif

#endif







