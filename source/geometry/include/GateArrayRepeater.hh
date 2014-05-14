/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateArrayRepeater_h
#define GateArrayRepeater_h 1

#include "globals.hh"

#include "GateVGlobalPlacement.hh"

class GateArrayRepeaterMessenger;

/*! \class  GateArrayRepeater
    \brief  The GateArrayRepeater models a repetition of an object in a 3D matrix,
    \brief  a pattern similar to the repetition of crystals in PET scanner blocks
    
    - GateArrayRepeater - by Daniel.Strul@iphe.unil.ch 
    
    - The array repeater uses five parameters: 
      	- the numbers of repetitions along X, Y and Z
	- a repetition vector
	- a centering flag
      Based on these parameters, it repeats an object at regular steps on a rectangular 3D-matrix,
      the step size in each direction being given by the corresponding coordinate of the repeat vector. 
      If the centering flag is on, the matrix is centered on the volume current position. If this flag
      is off, the first corner copy in the matrix is located at the volume current position.
    
*/      
class GateArrayRepeater  : public GateVGlobalPlacement
{
  public:
    //! Constructor
    //! The default repeater parameters are chosen so that, by default, the object is unchanged
    GateArrayRepeater(GateVVolume* itsObjectInserter,
      	      	       const G4String& itsName="cubicArray",
      	      	       G4int itsRepeatNumberX=1,G4int itsRepeatNumberY=1,G4int itsRepeatNumberZ=1,
      	      	       const G4ThreeVector& itsRepeatVector=G4ThreeVector(),
		       G4bool itsFlagAutoCenter=true);
    //! Destructor
    virtual ~GateArrayRepeater();

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
      	  { return m_repeatVector;}
     //! Get the number of repetitions along an axis (0=X, 1=Y, 2=Z)
     inline G4int GetRepeatNumber(size_t axis) 
      	  { return m_repeatNumber[axis];}
     //! Get the number of repetitions along X
     inline G4int GetRepeatNumberX() 
      	  { return GetRepeatNumber(0);}
     //! Get the number of repetitions along Y
     inline G4int GetRepeatNumberY() 
      	  { return GetRepeatNumber(1);}
     //! Get the number of repetitions along Z
     inline G4int GetRepeatNumberZ() 
      	  { return GetRepeatNumber(2);}
     //! Get the value of the centering flag
     inline G4bool GetAutoCenterFlag() 
      	  { return m_flagAutoCenter;}

     //! Set the repetition vector
     void SetRepeatVector(const G4ThreeVector& val) 
      	  { m_repeatVector = val; }
     //! Set the number of repetitions along an axis (0=X, 1=Y, 2=Z)
     void SetRepeatNumber(size_t axis, G4int number) 
      	  { m_repeatNumber[axis] = number; }
     //! Set the number of repetitions along X
     void SetRepeatNumberX(G4int number) 
      	  { SetRepeatNumber(0,number); }
     //! Set the number of repetitions along Y
     void SetRepeatNumberY(G4int number) 
      	  { SetRepeatNumber(1,number); }
     //! Set the number of repetitions along Z
     void SetRepeatNumberZ(G4int number) 
      	  { SetRepeatNumber(2,number); }
     //! Set the value of the centering flag
     void SetAutoCenterFlag(G4bool val)
      	  { m_flagAutoCenter = val; }

    //@}

  protected:
    //! \name repeater parameters
    //@{
    G4ThreeVector m_repeatVector;     	      	//!< Repetition vector
    G4int         m_repeatNumber[3];    	//!< Number of repetitions along the 3 axes
    G4bool    	  m_flagAutoCenter;   	      	//!< Centering flag
    //@}
    
    //! Messenger
    GateArrayRepeaterMessenger* m_Messenger; 

};

#endif

