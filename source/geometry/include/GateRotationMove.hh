/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GATEROTATIONMOVE_H
#define GATEROTATIONMOVE_H 1

#include "globals.hh"
#include "GateVGlobalPlacement.hh"

class GateRotationMoveMessenger;

/*! \class  GateRotationMove
    \brief  The GateRotationMove models a rotation movement    
    - GateRotationMove - by Daniel.Strul@iphe.unil.ch     
    - The movement equation modeled by this class is theta(t) = V x t 
      where theta(t) is the rotation angle around an axis at time t, and
      V is the angular rotation speed.      
*/      
class GateRotationMove  : public GateVGlobalPlacement
{
  public:
    //! Constructor
    //! The default movement parameters are chosen so that, by default, the rotation
    //! is around the Z axis whilst the rotation speed is zero
    GateRotationMove(GateVVolume* itsObjectInserter,
      	      	      	 const G4String& itsName,
     	      	      	 const G4ThreeVector& itsRotationAxis=G4ThreeVector(0.,0.,1.),
			 double itsVelocity=0.);
    //! Destructor
    virtual ~GateRotationMove();

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
    //! Get the rotation axis
    virtual inline const G4ThreeVector& GetRotationAxis() 
      	  { return m_rotationAxis;}
    //! Get the rotation speed
    virtual inline G4double GetVelocity() 
      	  { return m_velocity;}

    //! Set the rotation axis
    virtual inline void SetRotationAxis(const G4ThreeVector& val) 
      	  { m_rotationAxis = val;  }
    //! Set the rotation speed
    virtual inline void SetVelocity(double val) 
      	  { m_velocity = val;  }

    //! Get the value of the rotation angle that was last computed by PushMyPlacement()
    virtual inline G4double GetCurrentAngle() 
      	  { return m_currentAngle;}
    //@}


  private:
    //! \name movement parameters
    //@{
    G4ThreeVector m_rotationAxis;	  //!< Rotation axis (dimensionless vector)
    G4double m_velocity;      	      	  //!< Rotation velocity (angular speed)
    //@}
    
    //! Rotation angle that was last computed by PushMyPlacement()
    G4double m_currentAngle;

    //! Messenger
    GateRotationMoveMessenger* m_Messenger; 

};

#endif

