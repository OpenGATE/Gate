/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVPulseProcessor_h
#define GateVPulseProcessor_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GatePulse.hh"
#include "GateClockDependent.hh"

class GatePulseProcessorChain;

/*! \class  GateVPulseProcessor
    \brief  Abstract base-class for pulse-processor components of the digitizer
    
    - GateVPulseProcessor - by Daniel.Strul@iphe.unil.ch
    
    - GateVPulseProcessor is the abstract base-class for all pulse-processors. 
      The pulse-processors are pluggable modules that can be inserted into the
      digitizer module's list. 
      
    - Their role is to process a list of pulses (coming from the hit-convertor or 
      from another pulse-processor) and to return a new, processed list of pulses, 
      which can be fed to another pulse-processor or to the coincidence sorter
      for example.
    
    - When developping a new pulse-processor, one should:
      - develop the pulse-processing functions (see below);
      - develop a messenger for this pulse-processor;
      - add the new pulse-processor to the list of choices available in 
      	GatePulseProcessorChainMessenger.
    
    - To develop the pulse-processing functions, the developper has two options:
      - The easiest option is to re-use the pre-defined method ProcessPulseList(). 
      	This method processes an input pulse-list by repeatedly calling the method ProcessOnePulse()
	once for each input-pulse. The responsability of the developper then is to implement the pure virtual
	method ProcessOnePulse() so as to perform the required processing.
      - The other option is to overload the method ProcessPulseList() (if the pulse-processing 
      	sequential mechanism provided by ProcessPulseList() is not appropriate. 
	In that case, one should provide some dummy implementation (such as {;}) for ProcessOnePulse()
      	
      \sa GatePulseProcessorChainMessenger, GatePulse, GatePulseList
*/      
class GateVPulseProcessor : public GateClockDependent
{
  public:

    //! \name constructors and destructors
    //@{
    
    //! Constructs a new pulse-processor attached to a GateDigitizer
    GateVPulseProcessor(GatePulseProcessorChain* itsChain,
      	      	      	const G4String& itsName);

    virtual inline ~GateVPulseProcessor() {}  
    //@}


    //! \name pulse-processing functions
    //@{
    
    //! Default function for pulse-list processing
    //! This function reads an input pulse-list. It then calls sequantially the method
    //! ProcessOnePulse(), once for each of the input pulses. The method returns
    //! the output pulse-list resulting from this processing
    virtual GatePulseList* ProcessPulseList(const GatePulseList* inputPulseList);

    //! Pure virtual function for processing one input-pulse
    //! This function is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing must be incorporated into the output pulse-list
    virtual void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)=0;
    //@}

   
    //! \name getters and setters
    //@{

     inline GatePulseProcessorChain* GetChain()
       { return m_chain; }

   //@}

     //! Method overloading GateClockDependent::Describe()
     //! Print-out a description of the component
     //! Calls the pure virtual method DecribeMyself()
     virtual void Describe(size_t indent=0);
     
     //! Pure virtual method DecribeMyself()
     virtual void DescribeMyself(size_t indent=0) =0 ;
     
  protected:
    GatePulseProcessorChain* m_chain;
};


#endif

