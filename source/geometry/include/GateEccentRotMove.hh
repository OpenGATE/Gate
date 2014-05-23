/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateEccentRotMove_h
#define GateEccentRotMove_h 1

#include "GateVGlobalPlacement.hh"
#include "globals.hh"

#include "GateVGlobalPlacement.hh"
#include "G4Point3D.hh"
#include "G4Transform3D.hh"

class GateEccentRotMoveMessenger;

/*! \class  GateEccentRotMove
    \brief  The GateEccentRotMove models a movement where 
    \brief  - the volume is shifted (3D space) to its new eccentric position  
    \brief  - the volume orbits around the world O-Z axis
    \brief  This move is equivalent to translate the volume respect to r followed by applying an orbiting 
    \brief  move with an rotation axis passing trough O lab center and autorotation flag on.
    
    - GateEccentRotMove - by Jean-Marc. Vieira@iphe.unil.ch
    
    - The movement modeled by this class is an orbit, where the volume revolves
      around the O-Z lab frame axis at a constant speed. At any moment, each points of the 
      the volume remains at the same distance from the axis, along a direction whose angle 
      is theta(t) = V x t where V is the angular rotation speed and t the time.

    - Based of the orbiting (cf CLHEP lib. 3D transforms) move with an auto-rotation mode enabled, the object rotates at the same 
      time as it revolves.

      
*/      
class GateEccentRotMove  : public GateVGlobalPlacement
{
  public:
    //! Constructor
    //! The default movement parameters are chosen so that, by default, the rotation
    //! is around the Z axis whilst the rotation speed is zero
    //! no shifts by defauts 
    GateEccentRotMove(GateVVolume* itsObjectInserter,
		      const G4String& itsName="EccentRot",
		      const G4ThreeVector& itsShift=G4ThreeVector(0.,0.,0.),
		      double itsVelocity=0.);
    //! Destructor
    virtual ~GateEccentRotMove();

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

    //! Get the shift of eccentric rotation 
    inline const G4ThreeVector& GetShift() 
      	  { return m_shift;}

    //! Get the rotation speed
    inline G4double GetVelocity() 
      	  { return m_velocity;}

    //! Get the value of the rotation angle that was last computed by PushMyPlacement()
    virtual inline G4double GetCurrentAngle() 
      	  { return m_currentAngle;}


    //! Set the shift position in the transaxial plane (z=0 plan) regards to the OZ rotation axis
    void SetShift(const G4ThreeVector& val) 
      	  { m_shift = val; }
    //! Set the rotation speed
    void SetVelocity(double val) 
      	  { m_velocity = val;  }
    //@}

  private:
    //! \name movement parameters
    //@{
    G4ThreeVector m_shift;     	      	  //!< X,Y,Z shift coordinates to new positions 
    G4double m_velocity;     	      	  //!< Rotation velocity (angular speed)

    //@}
    
    //! Rotation angle that was last computed by PushMyPlacement()
    G4double m_currentAngle;

    //! Messenger
    GateEccentRotMoveMessenger* m_Messenger; 

};

#endif

