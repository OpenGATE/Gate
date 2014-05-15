/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePileup_h
#define GatePileup_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"

class GatePileupMessenger;
class GateOutputVolumeID;

/*! \class  GatePileup
    \brief  Pulse-processor modelling a pileup (maximum energy wins) of a crystal-block

    - The Pileup is parameterised by its 'depth': pulses will be summed up if their volume IDs
      are identical up to this depth. For instance, the default depth is 1: this means that
      pulses will be considered as taking place in a same block if the first two figures
      of their volume IDs are identical
    - A second parameter is added : the width of the pilup window

    - The class is largely inspired from the GateReadout class,
      but is aimed to work by time and not by event.

      \sa GateVPulseProcessor
*/
class GatePileup : public GateVPulseProcessor
{
  public:

    //! Constructs a new Pileup attached to a GateDigitizer
    GatePileup(GatePulseProcessorChain* itsChain,const G4String& itsName) ;

    //! Destructor
    virtual ~GatePileup() ;

    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the Pileup
    virtual void DescribeMyself(size_t indent);

    //! Returns the depth of the Pileup
    inline G4int GetDepth() const  	      	{ return m_depth; }

    //! Set the depth of the Pileup
    inline void  SetDepth(G4int aDepth)         { m_depth = aDepth; }

    //! Returns the time of the Pileup
    inline G4double GetPileup() const  	      	{ return m_pileup; }

    //! Set the time of the Pileup
    inline void  SetPileup(G4double aPileup)         { m_pileup = aPileup; }

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    virtual GatePulseList* ProcessPulseList(const GatePulseList* inputPulseList);
    virtual void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  private:
    //! The default is the one parameter that defines how a Pileup works:
    //! pulses will be summed up if their volume IDs are identical up to this depth.
    //! For instance, the default depth is 1: this means that pulses will be considered as
    //! taking place in a same block if the first two figures of their volume IDs are identical
    G4int m_depth;
    G4double m_pileup;
    GatePulseList m_waiting;

    GatePileupMessenger *m_messenger;	  //!< Messenger for this Pileup
};


#endif
