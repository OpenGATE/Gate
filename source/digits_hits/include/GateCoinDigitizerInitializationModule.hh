/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateCoinDigitizerInitializationModule
  	  	  Class that helps to initialize Digitizer, i. e. pass from HitsCollection to DigiCollection
  	  	  NB: Only DigiCollection could be modified and not HitsCollection which contains hits from SD
		05/2022 Olga.Kochebina@cea.fr
*/

#ifndef GateCoinDigitizerInitializationModule_h
#define GateCoinDigitizerInitializationModule_h 1

#include "G4VDigitizerModule.hh"
#include "GateCoincidenceDigi.hh"
#include "GateClockDependent.hh"
#include "GateCoinDigitizerInitializationModule.hh"
#include "globals.hh"
#include "GateCoincidenceDigitizer.hh"


class GateCoinDigitizerInitializationModule : public GateVDigitizerModule
{
public:

	GateCoinDigitizerInitializationModule(GateCoincidenceDigitizer *digitizer);
  ~GateCoinDigitizerInitializationModule();


  void Digitize() override;

  void DescribeMyself(size_t );

private:

  G4String m_colName;
  G4bool m_FirstEvent;
  std::vector<G4int> m_CDCIDs; // !-> Coincidence Digi Collection ID

  std::vector<G4String> m_inputNames;
  GateCoincidenceDigiCollection*  m_outputDigiCollection;
  GateCoincidenceDigi* m_outputDigi;
  std::vector<GateCoincidenceDigiCollection*>  m_inputDigiCollections;

  GateCoincidenceDigitizer* m_coinDigitizer;

  std::vector<G4double> test_times;

};

#endif
