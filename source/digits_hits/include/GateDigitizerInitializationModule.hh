/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateDigitizerInitializationModule
  	  	  Class that helps to initialize Digitizer, i. e. pass from HitsCollection to DigiCollection
  	  	  NB: Only DigiCollection could be modified and not HitsCollection which contains hits from SD
		05/2022 Olga.Kochebina@cea.fr
*/

#ifndef GateDigitizerInitializationModule_h
#define GateDigitizerInitializationModule_h 1

#include "G4VDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateDigitizerInitializationModule.hh"
#include "globals.hh"
#include "GateSinglesDigitizer.hh"


class GateDigitizerInitializationModule : public GateVDigitizerModule
{
public:

	GateDigitizerInitializationModule(GateSinglesDigitizer *digitizer);
  ~GateDigitizerInitializationModule();


  void Digitize() override;

  void DescribeMyself(size_t );

private:

  G4String m_colName;
  G4bool m_FirstEvent;
  G4int m_HCID;

  GateDigiCollection*  m_outputDigiCollection;
  GateSinglesDigitizer* m_digitizer;



};

#endif
