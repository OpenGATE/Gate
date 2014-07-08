/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCoincidenceMultiplesKiller_h
#define GateCoincidenceMultiplesKiller_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"
#include "GateObjectStore.hh"
#include "GateVCoincidencePulseProcessor.hh"

class GateCoincidenceMultiplesKillerMessenger;


class GateCoincidenceMultiplesKiller : public GateVCoincidencePulseProcessor
{
public:



  //! Destructor
  virtual ~GateCoincidenceMultiplesKiller() ;


  //! Constructs a new dead time attached to a GateDigitizer
  GateCoincidenceMultiplesKiller(GateCoincidencePulseProcessorChain* itsChain,
                                 const G4String& itsName);

public:

  //! Implementation of the pure virtual method declared by the base class GateClockDependent
  //! print-out the attributes specific of the MultiplesKiller
  virtual void DescribeMyself(size_t indent);

protected:

  /*! Implementation of the pure virtual method declared by the base class GateVCoincidencePulseProcessor*/
  GateCoincidencePulse* ProcessPulse(GateCoincidencePulse* inputPulse,G4int iPulse);



private:
  GateCoincidenceMultiplesKillerMessenger *m_messenger;    //!< Messenger
};


#endif
