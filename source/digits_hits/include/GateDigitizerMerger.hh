/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*! \class  GateDigitizerMerger
    \brief  GateDigitizerMerger merges  digis from several sensitive detectors

    - GateDigitizerMerger - by olga.kochebina@cea.fr 03/03/23

    \sa GateDigitizerMerger, GateDigitizerMergerMessenger
*/

#ifndef GateDigitizerMerger_h
#define GateDigitizerMerger_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateDigitizerMergerMessenger.hh"
#include "GateSinglesDigitizer.hh"


class GateDigitizerMerger : public GateVDigitizerModule
{
public:
  
  GateDigitizerMerger(GateSinglesDigitizer *digitizer, G4String name);
  ~GateDigitizerMerger();
  
  void Digitize() override;

  void AddInputCollection(const G4String& );
  void DescribeMyself(size_t indent );

protected:
  std::vector<G4String> m_names;
  std::vector<G4int> m_inputCollectionIDs;
  G4bool isFirstEvent;

private:
  GateDigi* m_outputDigi;

  GateDigitizerMergerMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;


};

#endif








