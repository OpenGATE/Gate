/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCoincidencePulseProcessorChain_h
#define GateCoincidencePulseProcessorChain_h 1

#include "globals.hh"
#include <vector>
#include <CLHEP/Random/RandFlat.h>
#include "GateModuleListManager.hh"

class GateDigitizer;
class GateCoincidencePulse;
class GateVCoincidencePulseProcessor;
class GateCoincidencePulseProcessorChainMessenger;
class GatePulseList;
class GateVSystem;

//    Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr, for the multi-system approach.

class GateCoincidencePulseProcessorChain : public GateModuleListManager
{
  public:
    GateCoincidencePulseProcessorChain(GateDigitizer* itsDigitizer,
    			    const G4String& itsOutputName);
    virtual ~GateCoincidencePulseProcessorChain();

     virtual void InsertProcessor(GateVCoincidencePulseProcessor* newChildProcessor);

     /*! \brief Virtual method to print-out a description of the object

	\param indent: the print-out indentation (cosmetic parameter)
     */    
     virtual void Describe(size_t indent=0);

     virtual void DescribeProcessors(size_t indent=0);
     virtual void ListElements();
     virtual GateVCoincidencePulseProcessor* FindProcessor(const G4String& name)
      	  { return (GateVCoincidencePulseProcessor*) FindElement(name); }
     virtual GateVCoincidencePulseProcessor* GetProcessor(size_t i)
      	  {return (GateVCoincidencePulseProcessor*) GetElement(i);}
     void ProcessCoincidencePulses();
     virtual size_t GetProcessorNumber()
      	  { return size();}
	  
     std::vector<G4String>& GetInputNames()
       { return m_inputNames; }
     const G4String& GetOutputName() const
       { return m_outputName; }
     const std::vector<GateCoincidencePulse*> MakeInputList() const;

     virtual inline GateVSystem* GetSystem() const
       { return m_system;}
     virtual inline void SetSystem(GateVSystem* aSystem)
       { m_system = aSystem; }
     
     // Next two methods were added for the multi-system approach
     virtual inline void SetSystem(G4String& inputName)
     {SetSystem(FindSystem(inputName));}
     virtual GateVSystem* FindSystem(G4String& inputName);
     
     void SetNoPriority(G4bool b){m_noPriority = b;}
  protected:
      GateCoincidencePulseProcessorChainMessenger*    m_messenger;
      GateVSystem *m_system;            //!< System to which the chain is attached
      G4String				   m_outputName;
      std::vector<G4String>                m_inputNames;
      G4bool         	      	           m_noPriority;
};

#endif

