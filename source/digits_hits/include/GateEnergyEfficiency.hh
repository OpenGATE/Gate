/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateEnergyEfficiency_h
#define GateEnergyEfficiency_h 1

#include "globals.hh"
#include <iostream>
#include <vector>

#include "GateVPulseProcessor.hh"

class GateEnergyEfficiencyMessenger;
class GateVDistribution;
class GateEnergyEfficiency : public GateVPulseProcessor
{
  public:

    GateEnergyEfficiency(GatePulseProcessorChain* itsChain,
		 const G4String& itsName=theTypeName) ;

    //! Destructor
    virtual ~GateEnergyEfficiency() ;

    virtual void DescribeMyself(size_t indent);
    void SetEfficiency(GateVDistribution* dist) {m_efficiency = dist;}
    GateVDistribution* GetEfficiency() const {return m_efficiency;}

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);
  private:
    GateVDistribution* m_efficiency;    	   //!< efficiency table
    GateEnergyEfficiencyMessenger *m_messenger;   //!< Messenger

    static const G4String& theTypeName;   //!< Default type-name for all efficiency

};


#endif
