/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateNoise_h
#define GateNoise_h 1

#include "globals.hh"
#include <iostream>
#include <vector>

#include "GateVPulseProcessor.hh"
#include "GatePulse.hh"

class GateNoiseMessenger;
class GateVDistribution;
class GateNoise : public GateVPulseProcessor
{
  public:

    GateNoise(GatePulseProcessorChain* itsChain,
		 const G4String& itsName=theTypeName) ;

    //! Destructor
    virtual ~GateNoise() ;


    //! Implementation of the pure virtual method declared by the base class GateDigitizerComponent
    //! print-out the attributes specific of the blurring
    virtual void DescribeMyself(size_t indent);
    void SetEnergyDistribution(GateVDistribution* energyDistrib) {m_energyDistrib=energyDistrib;}
    GateVDistribution* GetEnergyDistribution() const {return m_energyDistrib;}
    void SetDeltaTDistribution(GateVDistribution* deltaTDistrib) {m_deltaTDistrib=deltaTDistrib;}
    GateVDistribution* GetDeltaTDistribution() const {return m_deltaTDistrib;}

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list

    virtual GatePulseList* ProcessPulseList(const GatePulseList* inputPulseList);
    virtual void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);
  private:
    GateVDistribution* m_deltaTDistrib;  //! Delta arrival time distribution
    GateVDistribution* m_energyDistrib;  //! The energy distribution
    GatePulseList m_createdPulses;       //! Trans. pulse list
    GateNoiseMessenger *m_messenger;     //!< Messenger

    static const G4String& theTypeName;  //!< Default type-name for all efficiency
    G4double   	m_oldTime;                   //!< Time of last event
};


#endif
