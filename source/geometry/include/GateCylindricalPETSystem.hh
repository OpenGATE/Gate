/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCylindricalPETSystem_h
#define GateCylindricalPETSystem_h 1

#include "globals.hh"

#include "GateVSystem.hh"

class GateClockDependentMessenger;
class GateCylindricalPETSystemMessenger;

/*! \class  GateCylindricalPETSystem
    \brief  The GateCylindricalPETSystem models a ClearPET-like scanner
    
    - GateCylindricalPETSystem - by Daniel.Strul@iphe.unil.ch
    
    - A GateCylindricalPETSystem models a scanner that obeys to the convention that were defined
      by the ClearPET software group. It comprises a hierarchy of components ranging from the
      rsectors (detector panels, arranged in several rings) down to the layers (individual
      crystals of a phoswich column). 
      
    - This system overloads two methods of the GateVSystem base-class, Describe() and
      PrintToStream()

    - Beside the standard system methods, it also provides the method ComputeInternalRadius() 
      to compute the internal radius of the scanner
*/      
class GateCylindricalPETSystem : public GateVSystem
{
  public:
    //! Constructor
    GateCylindricalPETSystem(const G4String& itsName);
    //! Destructor
    virtual ~GateCylindricalPETSystem();


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

		void AddNewRSECTOR( G4String );
    
    //! Compute the internal radius of the crystal ring.
    virtual G4double ComputeInternalRadius();

   private:
		G4int   m_maxindex;
    G4int   m_maxrsectorID;
    std::map< G4String , G4int > m_rsectorID;
		GateCylindricalPETSystemMessenger* m_messenger2;
    GateClockDependentMessenger    	*m_messenger; 	//!< pointer to the system messenger
};

#endif

