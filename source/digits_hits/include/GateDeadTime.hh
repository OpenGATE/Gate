/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#ifndef GateDeadTime_h
#define GateDeadTime_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"
#include "GateObjectStore.hh"

class GateDeadTimeMessenger;


/*! \class  GateDeadTime
  \brief  Pulse-processor modelling a simple dead time discriminator.

  - GateDeadTime - by Luc.Simon@iphe.unil.ch

  - The method ProcessOnePulse of this class models a simple
  deadTime discriminator. User chooses value of dead time, mode
  (paralysable or not) and geometric level of application (crystal, module,...)

  \sa GateVPulseProcessor
  \sa GateVolumeID
  \sa GatePulseProcessorChainMessenger
*/

class GateDeadTime : public GateVPulseProcessor
{
public:

  //! Destructor
  virtual ~GateDeadTime() ;

  //! Check the validity of the volume name where the dead time will be applied
  void CheckVolumeName(G4String val);

  //! Constructs a new dead time attached to a GateDigitizer
  GateDeadTime(GatePulseProcessorChain* itsChain, const G4String& itsName);

public:

  //! Returns the deadTime
  unsigned long long int GetDeadTime() { return m_deadTime; }

  //! Set the deadTime
  void SetDeadTime(G4double val) { m_deadTime = (unsigned long long int)(val/picosecond); }

  //! Set the deadTime mode ; candidates : paralysable nonparalysable
  void SetDeadTimeMode(G4String val);
  //! Set the buffer mode ;
  void SetBufferMode(G4int val) { m_bufferMode=val; }
  //! Set the buffer mode ;
  void SetBufferSize(G4double val) { m_bufferSize=val; }
  //! Set the buffer mode ;

  //! Implementation of the pure virtual method declared by the base class GateClockDependent
  //! print-out the attributes specific of the deadTime
  virtual void DescribeMyself(size_t indent);

protected:

  /*! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    This methods processes one input-pulse
    It is is called by ProcessPulseList() for each of the input pulses
    The result of the pulse-processing is incorporated into the output pulse-list
    This method manages the updating of the "rebirth time table", the table of times when
    the detector volume will be alive again.
  */
  void ProcessOnePulse(const GatePulse* inputPulse, GatePulseList&  outputPulseList);

  //! To summarize it finds the number of elements of the different scanner levels
  void FindLevelsParams(GateObjectStore* anInserterStore);

private:
  G4String m_volumeName;  //!< Name of the volume where Dead time is applied
  G4int m_testVolume;     //!< equal to 1 if the volume name is valid, 0 else
  std::vector<int> numberOfComponentForLevel; //!< Table of number of element for each geometric level
  G4int numberOfHigherLevels ;  //!< number of geometric level higher than the one chosen by the user
  unsigned long long int m_deadTime; //!< DeadTime value
  // was :  G4String m_deadTimeMode;   //!< dead time mode : paralysable nonparalysable
  G4bool m_isParalysable;   //!< dead time mode : paralysable (true) nonparalysable (false) (modif. by D. Guez on 03/03/04)
  std::vector<unsigned long long int>  m_deadTimeTable;  //!< contains the "rebirth times". Alocated once at the first call.
  G4double m_bufferSize;  //!< contains the rebirth time.
  std::vector<double> m_bufferCurrentSize;  //!< contains the buffers sizes
  G4int m_bufferMode; //! 0 : DT during writing, 1 : DT if writing AND buffer full
  G4int m_init_done_run_id;
  GateDeadTimeMessenger *m_messenger;    //!< Messenger
};


#endif
