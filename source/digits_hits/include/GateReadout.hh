/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateReadout_h
#define GateReadout_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"

class GateReadoutMessenger;
class GateOutputVolumeID;

/*! \class  GateReadout
    \brief  Pulse-processor modelling a simple PMT readout (maximum energy wins) of a crystal-block

    - GateReadout - by Daniel.Strul@iphe.unil.ch

    - The readout is parameterised by its 'depth': pulses will be summed up if their volume IDs
      are identical up to this depth. For instance, the default depth is 1: this means that
      pulses will be considered as taking place in a same block if the first two figures
      of their volume IDs are identical

      \sa GateVPulseProcessor
*/
class GateReadout : public GateVPulseProcessor
{
  public:

    //! Constructs a new readout attached to a GateDigitizer
    GateReadout(GatePulseProcessorChain* itsChain,const G4String& itsName) ;

    //! Destructor
    virtual ~GateReadout() ;

    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the readout
    virtual void DescribeMyself(size_t indent);

    //! Returns the depth of the readout
    inline G4int GetDepth() const  	      	{ return m_depth; }

    //! Set the depth of the readout
    inline void  SetDepth(G4int aDepth)         { m_depth = aDepth; }

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  private:
    //! The default is the one parameter that defines how a readout works:
    //! pulses will be summed up if their volume IDs are identical up to this depth.
    //! For instance, the default depth is 1: this means that pulses will be considered as
    //! taking place in a same block if the first two figures of their volume IDs are identical
    G4int m_depth;

    GateReadoutMessenger *m_messenger;	  //!< Messenger for this readout
};


#endif
