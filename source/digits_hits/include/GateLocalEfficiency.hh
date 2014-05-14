/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GateLocalEfficiency_h
#define GateLocalEfficiency_h 1

#include "globals.hh"
#include <iostream>
#include <vector>

#include "GateVPulseProcessor.hh"

class GateLocalEfficiencyMessenger;
class GateVDistribution;
class GateLocalEfficiency : public GateVPulseProcessor
{
  public:

    GateLocalEfficiency(GatePulseProcessorChain* itsChain,
		 const G4String& itsName=theTypeName) ;

    //! Destructor
    virtual ~GateLocalEfficiency() ;

    virtual void DescribeMyself(size_t indent);
    void SetMode(size_t i,G4bool val);
    void SetEfficiency(GateVDistribution* dist) {m_efficiency = dist;}
    GateVDistribution* GetEfficiency() const {return m_efficiency;}

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);
    void ComputeSizes();
  private:
    std::vector<G4bool> m_enabled;    	  //!< is the level enabled
    GateVDistribution* m_efficiency;    	   //!< efficiency table
    GateLocalEfficiencyMessenger *m_messenger;   //!< Messenger

    static const G4String& theTypeName;   //!< Default type-name for all efficiency

};


#endif
