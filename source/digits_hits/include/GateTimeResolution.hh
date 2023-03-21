/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

/*! \class  GateTimeResolution
    \brief  GateTimeResolution does some dummy things with input digi
    to create output digi

    - GateTimeResolution - by Martin.Rey@epfl.ch (July 2003)

    \sa GateTimeResolution, GateTimeResolutionMessenger
*/

#ifndef GateTimeResolution_h
#define GateTimeResolution_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"
#include "GateSinglesDigitizer.hh"

#include "GateTimeResolutionMessenger.hh"


class GateTimeResolution : public GateVDigitizerModule
{
public:
  
  GateTimeResolution(GateSinglesDigitizer *digitizer, G4String name);
  ~GateTimeResolution();
  
  void Digitize() override;


  //! Returns the time resolution
  G4double GetFWHM()   	      { return m_fwhm; }

  //! Set the time resolution
  void SetFWHM(G4double val)   { m_fwhm = val;  }


  void DescribeMyself(size_t );

protected:
  G4double m_fwhm;

private:
  GateDigi* m_outputDigi;

  GateTimeResolutionMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;


};

#endif








