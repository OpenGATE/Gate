/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*! \class  GatePileup
    \brief  Digitizer module modeling a Pileup (maximum energy wins) of a crystal-block

    - The PileupOld is  by its 'depth': digis will be summed up if their volume IDs
      are identical up to this depth. For instance, the default depth is 1: this means that
      digis will be considered as taking place in a same block if the first two figures
      of their volume IDs are identical
    - A second parameter is added : the width of the pilup window

    - The class is largely inspired from the GateReadout class,
      but is aimed to work by time and not by event.

    OK: added to GND in Jan2023
*/
#ifndef GatePileup_h
#define GatePileup_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GatePileupMessenger.hh"
#include "GateSinglesDigitizer.hh"
#include "GateObjectStore.hh"


class GatePileup : public GateVDigitizerModule
{
public:
  
  GatePileup(GateSinglesDigitizer *digitizer, G4String name);
  ~GatePileup();
  
  void Digitize() override;

  //! Returns the depth of the Pileup
  inline G4int GetDepth() const  	      	{ return m_depth; }

  //! Set the depth of the Pileup
  inline void  SetDepth(G4int aDepth)         { m_depth = aDepth; }

  //! Returns the volume name of the Pileup
  inline G4String GetVolumeName() const  	      	{ return m_volumeName; }

  //! Set the volume name of the Pileup
  inline void  SetVolumeName(G4String name)         { m_volumeName = name; }


  //! Returns the time of the Pileup
  inline G4double GetPileup() const  	      	{ return m_Pileup; }

  //! Set the time of the Pileup
  inline void  SetPileup(G4double aPileup)         { m_Pileup = aPileup; }


  G4double ComputeStartTime(GateDigiCollection*) const;


//protected:
     //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
     //! This methods processes one input-pulse
     //! It is is called by ProcessPulseList() for each of the input pulses
     //! The result of the pulse-processing is incorporated into the output pulse-list
    // virtual GatePulseList* ProcessPulseList(const GatePulseList* inputPulseList);
     //virtual void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  void DescribeMyself(size_t );

protected:
  G4int m_depth;
  G4String m_volumeName;
  G4double m_Pileup;
  std::vector< GateDigi* >* m_waiting;
  G4bool m_firstEvent;


private:
  GateDigi* m_outputDigi;

  GatePileupMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;


};

#endif








