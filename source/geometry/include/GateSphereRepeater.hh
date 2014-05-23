/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSphereRepeater_h
#define GateSphereRepeater_h 1

#include "globals.hh"
#include "G4SystemOfUnits.hh"

#include "GateVGlobalPlacement.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

class GateSphereRepeaterMessenger;

/*! \class  GateSphereRepeater
    \brief  The GateSphereRepeater models a repetition of an object along a spherical pattern
    
    - GateSphereRepeater - by Delphine.Lazaro@imed.jussieu.fr
    
    - The sphere repeater uses five parameters: 
      	- the numbers of repetitions along axial axis (Z by default) and azimuthal axis (Y by default)
	- the radius of the sphere
	- the angles alpha and beta
      
    
*/      
class GateSphereRepeater  : public GateVGlobalPlacement
{
  public:
    //! Constructor
    //! The default repeater parameters are chosen so that, by default, the object is unchanged
    GateSphereRepeater(GateVVolume* itsObjectInserter,
      	      	       const G4String& itsName="sphere",
      	      	       G4int itsRepeatNumberWithPhi=1,
		       G4int itsRepeatNumberWithThete=1,
      	      	       G4double itsThetaAngle=0. * deg,
		       G4double itsPhiAngle=0. * deg,
		       G4double itsRadius=10. * mm,
		       G4bool itsFlagAutoCenter=true,
		       G4bool itsFlagAutoRotation=true );
    //! Destructor
    virtual ~GateSphereRepeater();

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
     //! Get the number of repetitions involving phi angle (latitude) 
     inline G4int GetRepeatNumberWithPhi() 
      	  { return m_repeatNumberWithPhi;}
     //! Get the number of repetitions involving theta angle (in the plane)
     inline G4int GetRepeatNumberWithTheta() 
      	  { return m_repeatNumberWithTheta;}
     //! Get the value of the centering flag
     inline G4bool GetAutoCenterFlag() 
      	  { return m_flagAutoCenter;}
     //! Get the value of the auto-rotation flag
     inline G4bool GetAutoRotation() const     	  { return m_flagAutoRotation;}
     //! Get the theta angle 
     inline G4double GetThetaAngle() const      { return m_thetaAngle;}
     //! Get the phi angle 
     inline G4double GetPhiAngle() const      { return m_phiAngle;}
     //! Get the sphere radius 
     inline G4double GetRadius() const      { return m_radius;}
     //! Get axial pitch 
     inline G4double GetAxialPitch() const      { return m_radius * sin(m_phiAngle);}


     
     //! Set the number of repetitions involving phi angle (latitude)
     void SetRepeatNumberWithPhi(G4int val) 
      	  { m_repeatNumberWithPhi = val; }
     //! Set the number of repetitions involving theta angle (in the plane)
     void SetRepeatNumberWithTheta(G4int val) 
      	  { m_repeatNumberWithTheta = val; }
     //! Set the value of the centering flag
     void SetAutoCenterFlag(G4bool val)
      	  { m_flagAutoCenter = val; }
     //! Set the value of the auto-rotation flag
     void SetAutoRotation(G4bool val)
          { m_flagAutoRotation = val;  }
     //! Set the theta angle
     void SetThetaAngle(G4double val) 
      	  { m_thetaAngle = val; }
     //! Set the phi angle
     void SetPhiAngle(G4double val) 
      	  { m_phiAngle = val; }
     //! Set the sphere radius
     void SetRadius(G4double val) 
      	  { m_radius = val; }

    //@}

  protected:
    //! \name repeater parameters
    //@{
    G4int         m_repeatNumberWithPhi;    	//!< Number of repetitions involving phi angle (latitude)
    G4int         m_repeatNumberWithTheta;      //!< Number of repetitions involving theta angle (in the plane)
    G4double      m_thetaAngle;                 //!< Angle theta of rotation along the ring, at each step
    G4double      m_phiAngle ;                  //!< Angle phi of rotation 
    G4double      m_radius    ;                 //!< Radius of the sphere along which the volumes are replicated 
    G4bool    	  m_flagAutoCenter;   	      	//!< Centering flag
    G4bool  	  m_flagAutoRotation; 	        //!< Auto-rotation flag
    //@}
    
    //! Messenger
    GateSphereRepeaterMessenger* m_Messenger; 

};

#endif

