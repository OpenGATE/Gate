/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateCoincidenceDeadTime.cc for more detals
  */


/*! \class  GateCoincidenceDeadTime
    \brief  GateCoincidenceDeadTime does some dummy things with input digi
    to create output digi

    - GateCoincidenceDeadTime - by name.surname@email.com

    \sa GateCoincidenceDeadTime, GateCoincidenceDeadTimeMessenger
*/

#ifndef GateCoincidenceDeadTime_h
#define GateCoincidenceDeadTime_h 1

#include "GateVDigitizerModule.hh"
#include "GateCoincidenceDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateCoincidenceDeadTimeMessenger.hh"
#include "GateCoincidenceDigitizer.hh"


class GateCoincidenceDeadTime : public GateVDigitizerModule
{
public:
  
  GateCoincidenceDeadTime(GateCoincidenceDigitizer *digitizer, G4String name);
  ~GateCoincidenceDeadTime();
  
  void Digitize() override;

  // *******implement your methods here
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
   void DescribeMyself(size_t indent);

protected:
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

private:
  GateCoincidenceDigi* m_outputDigi;

  GateCoincidenceDeadTimeMessenger *m_Messenger;

  GateCoincidenceDigiCollection*  m_OutputDigiCollection;

  GateCoincidenceDigitizer *m_digitizer;


};

#endif








