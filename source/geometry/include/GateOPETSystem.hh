/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateOPETSystem_h
#define GateOPETSystem_h 1

#include "globals.hh"

#include "GateVSystem.hh"

class GateClockDependentMessenger;

/*! \class  GateOPETSystem
    \brief  The GateOPETSystem models the UCLA's OPET system
    
    - GateOPETSystem - by Fernando R. Rannou
    
    - A GateOPETSystem models a scanner that obeys to the convention that were defined
      by the ClearPET software group. It comprises a hierarchy of components ranging from the
      rsectors (detector panels, arranged in several rings) down to the layers (individual
      crystals of a phoswich column). 
      
    - This system overloads two methods of the GateVSystem base-class, Describe() and
      PrintToStream()

    - Beside the standard system methods, it also provides the method ComputeInternalRadius() 
      to compute the internal radius of the scanner
*/      
class GateOPETSystem : public GateVSystem
{
  public:
    //! Constructor
    GateOPETSystem(const G4String& itsName);
    //! Destructor
    virtual ~GateOPETSystem();


    /*! \brief Method overloading the base-class virtual method Describe().
      	\brief This methods prints-out a description of the system, which is
	\brief optimised for creating LMF header files

	\param indent: the print-out indentation (cosmetic parameter)
    */    
    virtual void Describe(size_t indent=0);   	    

    /*! \brief Method overloading the base-class virtual method Describe().
      	\brief This methods prints out description of the system to a stream.
      	\brief It is essentially to be used by the class GateToLMF, but it may also be used by Describe()

	\param aStream: the output stream
	\param doPrintNumbers: tells whether we print-out the volume numbers in addition to their dimensions
    */    
    virtual void PrintToStream(std::ostream& aStream,G4bool doPrintNumbers);

    
    //! Compute the internal radius of the crystal ring.
    virtual G4double ComputeInternalRadius();

   private:
    GateClockDependentMessenger    	*m_messenger; 	//!< pointer to the system messenger
};

#endif

