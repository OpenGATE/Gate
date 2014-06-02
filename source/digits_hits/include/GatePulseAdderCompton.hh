/*----------------------
   OpenGATE Collaboration

   Daniel Strul <daniel.strul@iphe.unil.ch>
   JB Michaud <jbmichaud@videotron.ca>

   Copyright (C) 2002,2003 UNIL/IPHE, CH-1015 Lausanne
   Copyright (C) 2009 Universite de Sherbrooke

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GatePulseAdderCompton_h
#define GatePulseAdderCompton_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"
#include "G4Types.hh"

#include "GateVPulseProcessor.hh"

class GatePulseAdderComptonMessenger;

/*! \class  GatePulseAdder
    \brief  Pulse-processor for adding/grouping pulses per volume.

    - GatePulseAdder - by Daniel.Strul@iphe.unil.ch
	- Exact Compton kinematics changes by jbmichaud@videotron.ca

    - For each volume where there was one or more input pulse, we get exactly
      one output pulse, whose energy is the sum of all the input-pulse energies,
      and whose position is the centroid of the photonic input-pulse positions.
	  Electronic pulses energy is assigned to the proper photonic pulse (the last pulse encountered).
	  Wandering photo-electron are discarded, i.e. when no previous photonic interaction
	  has occured inside the volume ID. This is not EXACT.
	  The case when a photoelectron wanders and lands into a volume ID where

      \sa GateVPulseProcessor
*/
class GatePulseAdderCompton : public GateVPulseProcessor
{
  public:

    //! Constructs a new pulse-adder attached to a GateDigitizer
    GatePulseAdderCompton(GatePulseProcessorChain* itsChain,const G4String& itsName);

    //! Destructor
    virtual ~GatePulseAdderCompton();

    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the pulse adder
    virtual void DescribeMyself(size_t indent);

	//redefined because need to call repack() after last pulse
    //GatePulseList* ProcessPulseList(const GatePulseList* inputPulseList);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  private:
    GatePulseAdderComptonMessenger *m_messenger;     //!< Messenger

	//several functions needed for special processing of electronic pulses
	void PulsePushBack(const GatePulse* inputPulse, GatePulseList& outputPulseList);
	//void DifferentVolumeIDs(const GatePulse* InputPulse, GatePulseList& outputPulseList);
	//void repackLastVolumeID(GatePulseList& outputPulseList);

};


#endif
