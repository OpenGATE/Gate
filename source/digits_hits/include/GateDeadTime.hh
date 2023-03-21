/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*! \class  GateDeadTime
  \brief  Digitizer Module for a simple dead time discriminator.

  - GateDeadTime - by Luc.Simon@iphe.unil.ch

  - The method ProcessOnePulse of this class models a simple
  DeadTimeO discriminator. User chooses value of dead time, mode
  (paralysable or not) and geometric level of application (crystal, module,...)

  \sa GateVDigitizerModule
  \sa GateVolumeID


  - Added to New Digitizer GND by OK: January 2023
*/


#ifndef GateDeadTime_h
#define GateDeadTime_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "GateObjectStore.hh"

#include "globals.hh"

#include "GateDeadTimeMessenger.hh"
#include "GateSinglesDigitizer.hh"


class GateDeadTime : public GateVDigitizerModule
{
public:

   //! Constructs a new dead time attached to a GateDigitizer
  GateDeadTime(GateSinglesDigitizer *digitizer, G4String name);
  ~GateDeadTime();
  
  //! Check the validity of the volume name where the dead time will be applied
  void CheckVolumeName(G4String val);

  //! Returns the DeadTime
  unsigned long long int GetDeadTime() { return m_DeadTime; }

  //! Set the DeadTime
  void SetDeadTime(G4double val) { m_DeadTime = (unsigned long long int)(val/picosecond); }

  //! Set the DeadTime mode ; candidates : paralysable nonparalysable
  void SetDeadTimeMode(G4String val);
  //! Set the buffer mode ;
  void SetBufferMode(G4int val) { m_bufferMode=val; }
  //! Set the buffer mode ;
  void SetBufferSize(G4double val) { m_bufferSize=val; }
  //! Set the buffer mode ;

  //! Implementation of the pure virtual method declared by the base class GateClockDependent
  //! print-out the attributes specific of the DeadTime
  virtual void DescribeMyself(size_t indent);

  /*! Implementation of the pure virtual method declared by the base class GateVDigitizerModule
    This methods processes one input-pulse
    The result of the digitization is incorporated into the output digi collection
    This method manages the updating of the "rebirth time table", the table of times when
    the detector volume will be alive again.
  */
  void Digitize() override;
  
  //! To summarize it finds the number of elements of the different scanner levels
  void FindLevelsParams(GateObjectStore* anInserterStore);

protected:
  G4String m_volumeName;  //!< Name of the volume where Dead time is applied
  G4int m_testVolume;     //!< equal to 1 if the volume name is valid, 0 else
  std::vector<int> numberOfComponentForLevel; //!< Table of number of element for each geometric level
  G4int numberOfHigherLevels ;  //!< number of geometric level higher than the one chosen by the user
  unsigned long long int m_DeadTime; //!< DeadTime value
  // was :  G4String m_DeadTimeMode;   //!< dead time mode : paralysable nonparalysable
  G4bool m_isParalysable;   //!< dead time mode : paralysable (true) nonparalysable (false) (modif. by D. Guez on 03/03/04)
  std::vector<unsigned long long int>  m_DeadTimeTable;  //!< contains the "rebirth times". Alocated once at the first call.
  G4double m_bufferSize;  //!< contains the rebirth time.
  std::vector<double> m_bufferCurrentSize;  //!< contains the buffers sizes
  G4int m_bufferMode; //! 0 : DT during writing, 1 : DT if writing AND buffer full
  G4int m_init_done_run_id;

private:
  GateDigi* m_outputDigi;

  GateDeadTimeMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;


};

#endif








