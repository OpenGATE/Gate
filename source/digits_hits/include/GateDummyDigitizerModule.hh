/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateDummyDigitizerModule.cc for more detals
  */


/*! \class  GateDummyDigitizerModule
    \brief  GateDummyDigitizerModule does some dummy things with input digi
    to create output digi

    - GateDummyDigitizerModule - by name.surname@email.com

    \sa GateDummyDigitizerModule, GateDummyDigitizerModuleMessenger
*/

#ifndef GateDummyDigitizerModule_h
#define GateDummyDigitizerModule_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateDummyDigitizerModuleMessenger.hh"
#include "GateSinglesDigitizer.hh"


class GateDummyDigitizerModule : public GateVDigitizerModule
{
public:
  
  GateDummyDigitizerModule(GateSinglesDigitizer *digitizer, G4String name);
  ~GateDummyDigitizerModule();
  
  void Digitize() override;

  // *******implement your methods here
  void SetDummyParameter(const G4String& );
  
  void DummyMethod1(GateDigi *);
  void DummyMethod2(GateDigi *);

  void DescribeMyself(size_t );

protected:
  // *******implement your parameters here
  G4String   m_parameter;

private:
  GateDigi* m_outputDigi;

  GateDummyDigitizerModuleMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;


};

#endif








