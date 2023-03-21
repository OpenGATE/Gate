/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*! \class  GateAdderCompton

      \brief Digitizer Module for adding/grouping pulses per volume.

    - GatePulseAdder - by Daniel.Strul@iphe.unil.ch
	- Exact Compton kinematics changes by jbmichaud@videotron.ca

	   Copyright (C) 2002,2003 UNIL/IPHE, CH-1015 Lausanne
   	   Copyright (C) 2009 Universite de Sherbrooke

      OK: added to GND in Jan2023

    \sa GateAdderCompton, GateAdderComptonMessenger
*/

#ifndef GateAdderCompton_h
#define GateAdderCompton_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateAdderComptonMessenger.hh"
#include "GateSinglesDigitizer.hh"


class GateAdderCompton : public GateVDigitizerModule
{
public:
  
  GateAdderCompton(GateSinglesDigitizer *digitizer, G4String name);
  ~GateAdderCompton();
  
  void Digitize() override;
  GateDigi* CentroidMergeCompton(GateDigi *right, GateDigi *output);

  void DescribeMyself(size_t );

private:
  GateDigi* m_outputDigi;

  GateAdderComptonMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;


};

#endif








