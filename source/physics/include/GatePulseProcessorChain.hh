/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePulseProcessorChain_h
#define GatePulseProcessorChain_h 1

#include "globals.hh"
#include <vector>

#include "GateModuleListManager.hh"

class GateDigitizer;
class GateVPulseProcessor;
class GatePulseProcessorChainMessenger;
class GatePulseList;
class GateVSystem;

class GatePulseProcessorChain : public GateModuleListManager
{
  public:
    GatePulseProcessorChain(GateDigitizer* itsDigitizer,
    			    const G4String& itsOutputName);
    virtual ~GatePulseProcessorChain();

     virtual void InsertProcessor(GateVPulseProcessor* newChildProcessor);

     /*! \brief Virtual method to print-out a description of the object

	\param indent: the print-out indentation (cosmetic parameter)
     */    
     virtual void Describe(size_t indent=0);

     virtual void DescribeProcessors(size_t indent=0);
     virtual void ListElements();
     virtual GateVPulseProcessor* FindProcessor(const G4String& name)
      	  { return (GateVPulseProcessor*) FindElement(name); }
     virtual GateVPulseProcessor* GetProcessor(size_t i)
      	  {return (GateVPulseProcessor*) GetElement(i);}
     GatePulseList* ProcessPulseList();
     virtual size_t GetProcessorNumber()
      	  { return size();}
	  
     const G4String& GetInputName() const
       { return m_inputName; }
     void SetInputName(const G4String& anInputName)
       {  m_inputName = anInputName; }
     const G4String& GetOutputName() const
       { return m_outputName; }

     virtual inline GateVSystem* GetSystem() const
       { return m_system;}
     virtual inline void SetSystem(GateVSystem* aSystem)
       { m_system = aSystem; }

  protected:
      GatePulseProcessorChainMessenger*    m_messenger;
      GateVSystem *m_system;            //!< System to which the chain is attached
      G4String				   m_outputName;
      G4String                             m_inputName;
};

#endif

