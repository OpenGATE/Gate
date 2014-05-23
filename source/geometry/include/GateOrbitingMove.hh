/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateOrbitingMove_h
#define GateOrbitingMove_h 1

#include "globals.hh"

#include "GateVGlobalPlacement.hh"
#include "G4Point3D.hh"
#include "G4Transform3D.hh"

class GateOrbitingMoveMessenger;

/*! \class  GateOrbitingMove
    \brief  The GateRotationMove models a movement where the volume orbits
    \brief  around an axis
    
    - GateOrbitingMove - by Daniel.Strul@iphe.unil.ch 
    
    - The movement modeled by this class is an orbit, where the volume revolves
      around an axis at a constant speed. At any moment, the volume remains at
      the same distance from the axis, along a direction whose angle is theta(t) = V x t 
      where V is the angular rotation speed and t the time.

    - If the auto-rotation mode is enabled, the object rotates at the same time as it revolves.
      If the auto-rotation mode is disbaled, the object always keeps the same orientation.

      
*/      
class GateOrbitingMove  : public GateVGlobalPlacement
{
  public:
    //! Constructor
    //! The default movement parameters are chosen so that, by default, the rotation
    //! is around the Z axis whilst the rotation speed is zero
    //! By default, the auto-rotation mode is enabled
    GateOrbitingMove(GateVVolume* itsObjectInserter,
      	      	      	 const G4String& itsName,
     	      	      	 const G4Point3D& itsPoint1=G4Point3D(),
     	      	      	 const G4Point3D& itsPoint2=G4Point3D(0.,0.,1.),
			 double itsVelocity=0.,
			 G4bool itsFlagAutoRotation=true);
    //! Destructor
    virtual ~GateOrbitingMove();

    /*! \brief Implementation of the pure virtual method PushMyPlacement(), to compute
      	\brief a volume position and orientation as a function of time

	\param currentRotationMatrix: the rotation matrix that defines the current orientation of the volume
	\param currentPosition:       the vector that defines the current position of the volume
	\param aTime:                 the current time
	
	\return an object GatePlacement combining the new position and the new orientation of the volume
	        computed by the movement equation
    */    
     virtual void PushMyPlacements(const G4RotationMatrix& currentRotationMatrix,
	        	      	      	      const G4ThreeVector& currentPosition,
					      G4double aTime);
    /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
      	\brief a description of the movement

	\param indent: the print-out indentation (cosmetic parameter)
    */    
     virtual void DescribeMyself(size_t indent);
     
  public:
    //! \name getters and setters
    //@{
    //! Get the first point (origin) of the rotation axis
    inline const G4Point3D& GetPoint1() 
      	  { return m_point1;}
    //! Get the second point (end-point) of the rotation axis
    inline const G4Point3D& GetPoint2() 
      	  { return m_point2;}
    //! Get the rotation speed
    inline G4double GetVelocity() 
      	  { return m_velocity;}
    //! Get the auto-rotation mode flag
    inline G4bool GetAutoRotation() 
      	  { return m_flagAutoRotation;}

    //! Set the first point (origin) of the rotation axis
    void SetPoint1(const G4Point3D& val) 
      	  { m_point1 = val; }
    //! Set the second point (end-point) of the rotation axis
    void SetPoint2(const G4Point3D& val) 
      	  { m_point2 = val;  }
    //! Set the rotation speed
    void SetVelocity(double val) 
      	  {  m_velocity = val; }
    //! Set the auto-rotation mode flag
    void SetAutoRotation(G4bool val)
      	  { m_flagAutoRotation = val;  }

    //! Get the value of the rotation angle that was last computed by PushMyPlacement()
    inline G4double GetCurrentAngle() 
      	  { return m_currentAngle;}
    //@}

  private:
    //! \name movement parameters
    //@{
    G4Point3D m_point1;     	      	  //!< First point (origin) of the rotation axis
    G4Point3D m_point2;     	      	  //!< econd point (end-point) of the rotation axis
    G4double m_velocity;     	      	  //!< Rotation velocity (angular speed)
    G4bool   m_flagAutoRotation;      	  //!< Auto-rotation mode flag
    //@}
    
    //! Rotation angle that was last computed by PushMyPlacement()
    G4double m_currentAngle;

    //! Messenger
    GateOrbitingMoveMessenger* m_Messenger; 

};

#endif

