/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCoincidenceBuffer_h
#define GateCoincidenceBuffer_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVCoincidencePulseProcessor.hh"

class GateCoincidenceBufferMessenger;


class GateCoincidenceBuffer : public GateVCoincidencePulseProcessor
{
public:
  typedef unsigned long long int  buffer_t;



  //! Destructor
  virtual ~GateCoincidenceBuffer() ;


  //! Constructs a new dead time attached to a GateDigitizer
  GateCoincidenceBuffer(GateCoincidencePulseProcessorChain* itsChain,
			     const G4String& itsName);

public:

  void SetBufferSize(buffer_t val)   { m_bufferSize = val;}
  void SetReadFrequency(G4double val)   { m_readFrequency = val;}
//  void SetDoModifyTime(G4bool val)   { m_doModifyTime = val;}
  void SetMode(G4int val)   { m_mode = val;}


  //! Implementation of the pure virtual method declared by the base class GateClockDependent
  //! print-out the attributes specific of the deadTime
  virtual void DescribeMyself(size_t indent);

protected:

  /*! Implementation of the pure virtual method declared by the base class GateVCoincidencePulseProcessor*/
  GateCoincidencePulse* ProcessPulse(GateCoincidencePulse* inputPulse,G4int iPulse);


private:
  buffer_t m_bufferSize;
  buffer_t m_bufferPos;
  buffer_t m_oldClock;
  G4double m_readFrequency;
//  G4bool   m_doModifyTime;
  G4int    m_mode;
  GateCoincidenceBufferMessenger *m_messenger;    //!< Messenger
};


#endif
