/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateCoincidenceTimeDiffSelector_h
#define GateCoincidenceTimeDiffSelector_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

//#include "GateVPulseProcessor.hh"
#include "GateObjectStore.hh"

#include "GateVDigitizerModule.hh"
#include "GateCoincidenceDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"


#include "GateDigi.hh"

#include "GateCoincidenceTimeDiffSelectorMessenger.hh"
#include "GateCoincidenceDigitizer.hh"
class GateCoincidenceTimeDiffSelectorMessenger;


class GateCoincidenceTimeDiffSelector : public GateVDigitizerModule
{
public:

	GateCoincidenceTimeDiffSelector(GateCoincidenceDigitizer *digitizer, G4String name);
	  ~GateCoincidenceTimeDiffSelector();

  //! Destructor
  //virtual ~GateCoincidenceTimeDiffSelector() ;


  //! Constructs a new dead time attached to a GateDigitizer
  //GateCoincidenceTimeDiffSelector(GateCoincidencePulseProcessorChain* itsChain,
                                  //const G4String& itsName);
void Digitize() override;
public:

  //! Returns the TimeDiffSelector
  G4double GetMinTime() const {return m_minTime;}
  G4double GetMaxTime() const {return m_maxTime;}

  //! Set the TimeDiffSelector
  void SetMinTime(G4double val) { m_minTime = val;}
  void SetMaxTime(G4double val) { m_maxTime = val;};

  //! Implementation of the pure virtual method declared by the base class GateClockDependent
  //! print-out the attributes specific of the TimeDiffSelector
  virtual void DescribeMyself(size_t indent);

protected:

  /*! Implementation of the pure virtual method declared by the base class GateVCoincidencePulseProcessor*/
  //GateCoincidencePulse* ProcessPulse(GateCoincidencePulse* inputPulse,G4int iPulse);



private:
  G4double m_minTime; //!< TimeDiffSelector value
  G4double m_maxTime;  //!< contains the rebirth time.
  GateCoincidenceTimeDiffSelectorMessenger *m_messenger;    //!< Messenger
  GateCoincidenceDigi* m_outputDigi;

   GateCoincidenceDigiCollection*  m_OutputDigiCollection;

   GateCoincidenceDigitizer *m_digitizer;
};


#endif
