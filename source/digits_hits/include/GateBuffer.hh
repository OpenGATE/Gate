/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateBuffer_h
#define GateBuffer_h 1

#include "globals.hh"
#include <iostream>
#include <vector>

#include "GateVPulseProcessor.hh"
#include "GatePulse.hh"

class GateBufferMessenger;
class GateVDistribution;
class GateBuffer : public GateVPulseProcessor
{
  public:
    typedef unsigned long long int  buffer_t;

    GateBuffer(GatePulseProcessorChain* itsChain,
		 const G4String& itsName=theTypeName) ;

    //! Destructor
    virtual ~GateBuffer() ;


    //! Implementation of the pure virtual method declared by the base class GateDigitizerComponent
    //! print-out the attributes specific of the blurring
    void SetBufferSize(buffer_t val)   { m_bufferSize = val;}
    void SetReadFrequency(G4double val)   { m_readFrequency = val;}
    void SetDoModifyTime(G4bool val)   { m_doModifyTime = val;}
    void SetMode(G4int val)   { m_mode = val;}
    void SetDepth(size_t depth);
    virtual void DescribeMyself(size_t indent);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list

    virtual void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);
  private:
    buffer_t m_bufferSize;
    std::vector<buffer_t> m_bufferPos;
    buffer_t m_oldClock;
    G4double m_readFrequency;
    G4bool   m_doModifyTime;
    G4int    m_mode;
    std::vector<G4bool> m_enableList;
    GateBufferMessenger *m_messenger;     //!< Messenger

    static const G4String& theTypeName;  //!< Default type-name for all buffers
};


#endif
