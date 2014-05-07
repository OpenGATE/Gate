/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateLinearRepeater_h
#define GateLinearRepeater_h 1

#include "globals.hh"

#include "GateVGlobalPlacement.hh"

class GateVVolume;
class GateLinearRepeaterMessenger;

/*! \class  GateLinearRepeater
    \brief  The GateLinearRepeater models a repetition of an object along a line,
    \brief  a pattern similar to the repetition of the scanner rings in a PET scanner
    
    - GateLinearRepeater - by Daniel.Strul@iphe.unil.ch 
    
    - The linear repeater uses three parameters: 
      	- a number of repetitions 
	- a repetition vector
	- a centering flag
      Based on these parameters, it repeats an object at regular steps in a line,
      the step size being given by the repeat vector. If the centering flag is on, the 
      series of copies is centered on the volume current position. If this flag
      is off, the first copy in the series is located at the volume current position.
    
*/      
class GateLinearRepeater  : public GateVGlobalPlacement
{
  public:
    //! Constructor
    //! The default repeater parameters are chosen so that, by default, the object is unchanged
    GateLinearRepeater(GateVVolume* itsObjectInserter,
      	      	       const G4String& itsName="linear",
      	      	       G4int itsRepeatNumber=1,
      	      	       const G4ThreeVector& itsRepeatVector=G4ThreeVector(),
		       G4bool itsFlagAutoCenter=true);
    //! Destructor
    virtual ~GateLinearRepeater();

  public:
    /*! \brief Implementation of the pure virtual method PushMyPlacements(), to compute
      	\brief the position and orientation of all copies as a function of time. The series
	\brief of placements thus obtained is placed into the repeater placement queue.

	\param currentRotationMatrix: the rotation matrix that defines the current orientation of the volume
	\param currentPosition:       the vector that defines the current position of the volume
	\param aTime:                 the current time
	
    */    
     virtual void PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
	        	      	   const G4ThreeVector& currentPosition,
			      	   G4double aTime);
    /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
      	\brief a description of the repeater

	\param indent: the print-out indentation (cosmetic parameter)
    */    
     virtual void DescribeMyself(size_t indent);
     
  public:
    //! \name getters and setters
    //@{
     //! Get the repetition vector
     inline const G4ThreeVector& GetRepeatVector() 
      	  { return m_repeatVector; }
     //! Get the number of repetitions
     inline G4int GetRepeatNumber() 
      	  { return m_repeatNumber;}
     //! Get the value of the centering flag
     inline G4bool GetAutoCenterFlag() 
      	  { return m_flagAutoCenter;}
 
     //! Set the repetition vector
     void SetRepeatVector(const G4ThreeVector& val) 
      	  { m_repeatVector = val;  }
     //! Set the number of repetitions
     void SetRepeatNumber(G4int val) 
      	  { m_repeatNumber = val;  }
     //! Set the value of the centering flag
     void SetAutoCenterFlag(G4bool val)
      	  { m_flagAutoCenter = val;    }
 
    //@}

  protected:
    //! \name repeater parameters
    //@{
    G4ThreeVector m_repeatVector;     	//!< Repetition vector
    G4int         m_repeatNumber;     	//!< Number of repetitions
    G4bool    	  m_flagAutoCenter;   	//!< Centering flag
    //@}
    
    //! Messenger
    GateLinearRepeaterMessenger* m_Messenger; 
};

#endif

