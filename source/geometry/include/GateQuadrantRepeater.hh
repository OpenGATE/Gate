/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateQuadrantRepeater_h
#define GateQuadrantRepeater_h 1

#include "globals.hh"
#include "G4SystemOfUnits.hh"

#include "GateVGlobalPlacement.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

class GateQuadrantRepeaterMessenger;

/*! \class  GateQuadrantRepeater
    \brief  The GateQuadrantRepeater models a repetition of an object in a triangle-like
    \brief  pattern similar to the one used in Derenzo phantoms (this repeater was
    \brief specifically designed to allow the modelling of such phantoms)
    
    - GateQuadrantRepeater - by Daniel.Strul@iphe.unil.ch (Oct 2002)
    
    - The quadrant repeater uses three parameters: 
      	- a number of lines 
	- an orientation angle (in the XY plane)
	- a copy spacing
      Based on these parameters, it repeats an object in the XY plane in a series of 
      copy lines. The orientation angle gives the direction of line-replication,
      each line being orthogonal to this direction. The number of copies increases 
      from one line to the next one. The repetition vectors both between lines and
      between copies on a line are computed so that the copy spacing is identical 
      between each copy and its nearest neighbours. The resulting pattern is an
      equilateral triangle.
    
    - A fourth optional paramter allows to reject positions beyond a limit
*/      
class GateQuadrantRepeater  : public GateVGlobalPlacement
{
  public:
    //! Constructor
    //! The default repeater parameters are chosen so that, by default, the object is unchanged
    GateQuadrantRepeater(GateVVolume* itsObjectInserter,
      	      	       const G4String& itsName="linear",
      	      	       G4int itsLineNumber=1,
      	      	       G4double itsOrientation=0.,
		       G4double itsCopySpacing=0.);
    //! Destructor
    virtual ~GateQuadrantRepeater();

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
     //! Set the number of repetition lines
     inline void SetLineNumber(G4int aLineNumber) 
      	  { m_lineNumber = aLineNumber; }
     //! Get the number of repetition lines
     inline G4int GetLineNumber()  const
      	  { return m_lineNumber;}
 
     //! Set the orientation
     void SetOrientation(G4double val) 
      	  { m_orientation = val;  ComputeRepetitionVectors(); }
     //! Get the line-spacing vector
     inline G4double GetOrientation()  const
      	  {return m_orientation;  }

     //! Set the copy spacing
     inline void SetCopySpacing(G4double val) 
      	  { m_copySpacing = val; ComputeRepetitionVectors(); }
     //! Get the copy spacing
     inline G4double GetCopySpacing() const
      	  { return m_copySpacing;}
 
     //! Set the maximum range
     inline void SetMaxRange(G4double val) 
      	  { m_maxRange = val; }
     //! Get the maximum range
     inline G4double GetMaxRange() const
      	  { return m_maxRange;}
 
    //@}

    //! Compute the new value of the orthogonal repeat vector
    //! (this method must be caled whenever the line-spacing vector changes)
    void ComputeRepetitionVectors()
    {
      m_lineSpacingVector      = G4ThreeVector( cos(m_orientation-30.*deg),sin(m_orientation-30.*deg),0.)* m_copySpacing ;
      m_orthogonalRepeatVector = G4ThreeVector( cos(m_orientation+90.*deg),sin(m_orientation+90.*deg),0.)* m_copySpacing ;
    }

  protected:
    //! \name repeater parameters
    //@{
    G4int         m_lineNumber; 	  //!< Number of lines
    G4double  	  m_orientation;	  //!< line-replication direction
    G4double  	  m_copySpacing;      	  //!< Spacing between adjacent copies
    G4double  	  m_maxRange;      	  //!< Maximum distance from a copy to the original position
    //@}
    
    //! \name repeater vectors, derived from the parameters by ComputeRepetitionVectors()
    //@{
    G4ThreeVector m_lineSpacingVector; 	      //!< Displacement vector between adjacent lines
    G4ThreeVector m_orthogonalRepeatVector;   //!< Displacement vector between adajacent copies on a line
    //@}
    
    //! Messenger
    GateQuadrantRepeaterMessenger* m_Messenger; 
};

#endif

