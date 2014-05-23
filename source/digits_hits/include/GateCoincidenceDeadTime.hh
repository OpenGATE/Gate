/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCoincidenceDeadTime_h
#define GateCoincidenceDeadTime_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"
#include "GateObjectStore.hh"
#include "GateVCoincidencePulseProcessor.hh"

class GateCoincidenceDeadTimeMessenger;


class GateCoincidenceDeadTime : public GateVCoincidencePulseProcessor
{
public:



  //! Destructor
  virtual ~GateCoincidenceDeadTime() ;


  //! Constructs a new dead time attached to a GateDigitizer
  GateCoincidenceDeadTime(GateCoincidencePulseProcessorChain* itsChain,
                          const G4String& itsName);

public:

  //! Returns the deadTime
  unsigned long long int GetDeadTime() {return m_deadTime;}

  //! Set the deadTime
  void SetDeadTime(G4double val)   { m_deadTime = (unsigned long long int )(val/picosecond);}

  //! Set the deadTime mode ; candidates : paralysable nonparalysable
  void SetDeadTimeMode(G4String val);

  //! Set the buffer mode ;
  void SetBufferMode(G4int val){m_bufferMode=val;}
  //! Set the buffer mode ;
  void SetBufferSize(G4double val){m_bufferSize=val;}
  //! Set the way to treat events ;
  void SetConserveAllEvent(G4bool val){m_conserveAllEvent=val;}

  //! Implementation of the pure virtual method declared by the base class GateClockDependent
  //! print-out the attributes specific of the deadTime
  virtual void DescribeMyself(size_t indent);

protected:

  /*! Implementation of the pure virtual method declared by the base class GateVCoincidencePulseProcessor*/
  GateCoincidencePulse* ProcessPulse(GateCoincidencePulse* inputPulse,G4int iPulse);



private:
  unsigned long long int m_deadTime; //!< DeadTime value
  G4bool m_isParalysable;   //!< dead time mode : paralysable (true) nonparalysable (false) (modif. by D. Guez on 03/03/04)
  unsigned long long int m_rebirthTime;  //!< contains the rebirth time.
  G4double m_bufferSize;  //!< contains the rebirth time.
  G4double m_bufferCurrentSize;  //!< contains the rebirth time.
  G4int   m_bufferMode; //! 0 : DT during writing, 1 : DT if writing AND buffer full
  G4int m_oldEv1;
  G4int m_oldEv2;
  G4String m_oldName;
  G4bool m_conserveAllEvent;
  G4bool m_wasTaken;
  GateCoincidenceDeadTimeMessenger *m_messenger;    //!< Messenger
};


#endif
