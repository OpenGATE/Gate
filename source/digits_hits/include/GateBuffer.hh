/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022


/*! \class  GateBuffer
    \brief  GateBuffer mimics the effect of limited transfer rate

    5/12/2023 added to GND by kochebina@cea.fr

    \sa GateBuffer, GateBufferMessenger
*/

#ifndef GateBuffer_h
#define GateBuffer_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateBufferMessenger.hh"
#include "GateSinglesDigitizer.hh"


class GateBuffer : public GateVDigitizerModule
{
public:

	typedef unsigned long long int  Buffer_t;

  GateBuffer(GateSinglesDigitizer *digitizer, G4String name);
  ~GateBuffer();
  
  void Digitize() override;

  //! Implementation of the pure virtual method declared by the base class GateDigitizerComponent
  //! print-out the attributes specific of the blurring
  void SetBufferSize(Buffer_t val)   { m_BufferSize = val;}
  void SetReadFrequency(G4double val)   { m_readFrequency = val;}
  void SetDoModifyTime(G4bool val)   { m_doModifyTime = val;}
  void SetMode(G4int val)   { m_mode = val;}
  void SetDepth(size_t depth);

  void DescribeMyself(size_t );

protected:
  Buffer_t m_BufferSize;
  std::vector<Buffer_t> m_BufferPos;
  Buffer_t m_oldClock;
  G4double m_readFrequency;
  G4bool   m_doModifyTime;
  G4int    m_mode;
  std::vector<G4bool> m_enableList;

private:
  GateDigi* m_outputDigi;

  GateBufferMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;

  static const G4String& theTypeName;  //!< Default type-name for all Buffers

};

#endif








