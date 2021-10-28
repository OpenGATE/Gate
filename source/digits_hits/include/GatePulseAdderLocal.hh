/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GatePulseAdderLocal_h
#define GatePulseAdderLocal_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"
#include "GateMaps.hh"
#include "GatePulseAdder.hh"

class GatePulseAdderLocalMessenger;

/*! \class  GatePulseAdderLocal
    \brief  Pulse-processor for adding/grouping pulses per volume.

    - For each volume where there was one or more input pulse, we get exactly
      one output pulse, whose energy is the sum of all the input-pulse energies,
      and whose position is the centroid of the input-pulse positions by default. If no optiosn is selected

      \sa GateVPulseProcessor
*/



class GatePulseAdderLocal : public GateVPulseProcessor
{
  public:

    //! Constructs a new pulse-adder attached to a GateDigitizer
    GatePulseAdderLocal(GatePulseProcessorChain* itsChain,const G4String& itsName);

    //! Destructor
    virtual ~GatePulseAdderLocal();

    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the pulse adder
    virtual void DescribeMyself(size_t indent);
    void SetPositionPolicy(G4String & name, const G4String& policy);

    G4int chooseVolume(G4String val);
  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);


  private:
    GatePulseAdderLocalMessenger *m_messenger;     //!< Messenger

    //-----
    struct param {
         position_policy_t   m_positionPolicy;
    };
    param m_param;                                 //!<
    G4String m_name;                               //! Name of the volume
    GateMap<G4String,param> m_table ;  //! Table which contains the names of volume with their characteristics
    GateMap<G4String,param> ::iterator im;  //! iterator of the gatemap



};


#endif
