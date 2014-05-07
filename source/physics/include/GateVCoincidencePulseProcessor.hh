/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVCoincidencePulseProcessor_h
#define GateVCoincidencePulseProcessor_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateCoincidencePulse.hh"
#include "GateClockDependent.hh"

class GateCoincidencePulseProcessorChain;

/*! \class  GateVCoincidencePulseProcessor
    \brief  Abstract base-class for coincidence pulse-processor components of the digitizer
    
    - GateVCoincidencePulseProcessor - by davguez@yahoo.fr
      Mostly made as a simple copy-paste of the GateVPulseProcessor class, so
      take a look to this class' doc for more details...
    
*/      
class GateVCoincidencePulseProcessor : public GateClockDependent
{
  public:

    //! \name constructors and destructors
    //@{
    
    //! Constructs a new pulse-processor attached to a GateDigitizer
    GateVCoincidencePulseProcessor(GateCoincidencePulseProcessorChain* itsChain,
      	      	      	const G4String& itsName);

    virtual inline ~GateVCoincidencePulseProcessor() {}
    //@}


    //! \name pulse-processing functions
    //@{
    
    virtual GateCoincidencePulse* ProcessPulse(GateCoincidencePulse* inputPulse,G4int iPulse)=0;

    //@}

   
    //! \name getters and setters
    //@{

     inline GateCoincidencePulseProcessorChain* GetChain()
       { return m_chain; }
     //mhadi_add[
     virtual inline bool IsTriCoincProcessor() const { return 0; } 
     virtual void CollectSingles() {}
     //mhadi_add]
   //@}

     //! Method overloading GateClockDependent::Describe()
     //! Print-out a description of the component
     //! Calls the pure virtual method DecribeMyself()
     virtual void Describe(size_t indent=0);
     
     //! Pure virtual method DecribeMyself()
     virtual void DescribeMyself(size_t indent=0) =0 ;
     
  protected:
/*    bool                                m_isTriCoincProc;*/
    GateCoincidencePulseProcessorChain* m_chain;
};


#endif

