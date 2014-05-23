/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GateSystemFilter_h
#define GateSystemFilter_h 1
#include "globals.hh"
#include "GateVPulseProcessor.hh"

class GateSystemFilterMessenger;

/*! \class GateSystemFilter
    \brief Pulse processor module to separate between pulses arriving from systems in multi-system approach

    - GateSystemFilter - Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr
 */

class GateSystemFilter : public GateVPulseProcessor
{
  public:

    GateSystemFilter(GatePulseProcessorChain* itsChain,const G4String& itsName);

    virtual ~GateSystemFilter();

    inline G4String GetSystemName()                                  { return m_systemName; }
    inline void SetSystemName(G4String systemName)                   {m_systemName = systemName ;}
    void SetSystemToItsChain();

    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the system filter
    virtual void DescribeMyself(size_t indent);

  protected:
     GatePulseList* ProcessPulseList(const GatePulseList* inputPulseList);
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  private:
    GateSystemFilterMessenger *m_messenger;     //Messenger
    G4String m_systemName;
};

#endif
