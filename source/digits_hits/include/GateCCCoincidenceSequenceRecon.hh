/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateCCCoincidenceSequenceRecon_h
#define GateCCCoincidenceSequenceRecon_h 1

#include "globals.hh"
#include <iostream>
#include <vector>

#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include <algorithm>

#include "G4ThreeVector.hh"
#include "GateVPulseProcessor.hh"
#include "GateObjectStore.hh"
#include "GateVCoincidencePulseProcessor.hh"

class GateCCCoincidenceSequenceReconMessenger;

typedef enum {kSinglesTime,
              kLowestEnergyFirst,
              kRandomly,
              kSpatially,
              kRevanC_CSR,
             } sequence_policy_t;


class GateCCCoincidenceSequenceRecon : public GateVCoincidencePulseProcessor
{
public:
    //! Destructor
    virtual ~GateCCCoincidenceSequenceRecon() ;

    //! Constructs a coincidence sorter attached to a GateDigitizer
    GateCCCoincidenceSequenceRecon(GateCoincidencePulseProcessorChain* itsChain,const G4String& itsName);

public:
    void SetSequencePolicy(const G4String& policy);

    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the sequence reconstruction
    virtual void DescribeMyself(size_t indent);

protected:

    /*! Implementation of the pure virtual method declared by the base class GateVCoincidencePulseProcessor*/
    GateCoincidencePulse* ProcessPulse(GateCoincidencePulse* inputPulse, G4int);



private:
    const static int INVALID_Qf=100000;
    double computeGeomScatteringAngleError( GateCoincidencePulse coincP);

    GateCCCoincidenceSequenceReconMessenger *m_messenger;    //!< Messenger

    sequence_policy_t m_sequencePolicy;
    unsigned seed;
};


#endif
