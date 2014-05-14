/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateTranslationMove_h
#define GateTranslationMove_h 1

#include "globals.hh"

#include "GateVGlobalPlacement.hh"

class GateTranslationMoveMessenger;

/*! \class  GateTranslationMove
    \brief  The GateTranslationMove models a constant speed translation movement
    
    - GateTranslationMove - by Daniel.Strul@iphe.unil.ch
    
    - The movement equation modeled by this class is dM(t) = V x t 
      where dM(t) is the translation vector at time t, V is the translation velocity, 
      and t is the time
      
*/      
class GateTranslationMove  : public GateVGlobalPlacement
{
  public:
    //! Constructor
    //! The default movement parameters are chosen so that, by default, the movement is null
    GateTranslationMove(GateVVolume* itsObjectInserter,
      	      	      	      	  const G4String& itsName,
				  const G4ThreeVector& itsVelocity=G4ThreeVector());
    //! Destructor
    virtual ~GateTranslationMove();

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
    //! Get the translation velocity
    inline const G4ThreeVector& GetVelocity() 
      	  { return m_velocity;}
    //! Set the translation velocity
    void SetVelocity(const G4ThreeVector& val) 
      	  { m_velocity = val;  }

    //! Get the value of the translation vector that was last computed by PushMyPlacement()
    inline const G4ThreeVector& GetCurrentTranslation() 
      	  { return m_currentTranslation;}
    //@}

  private:
    //! \name movement parameters
    //@{
    G4ThreeVector m_velocity; 	//!< Translation velocity
    //@}
    
    //! Translation vector that was last computed by PushMyPlacement()
    G4ThreeVector m_currentTranslation;
    
    //! Messenger
    GateTranslationMoveMessenger* m_Messenger; 

};

#endif

