/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GateVolumePlacement_h
#define GateVolumePlacement_h 1

#include "globals.hh"
#include "G4SystemOfUnits.hh"
#include "G4Point3D.hh"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"

#include "GateVGlobalPlacement.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

//--------------------------------------------------------------------
class GateVolumePlacementMessenger;

/*! \class  GateVolumePlacement
  \brief  The GateVolumePlacement models a static combination of a translation and of a rotation
    
  - GateVolumePlacement - by Daniel.Strul@iphe.unil.ch 
    
  - The GateVolumePlacement defines both a static translation (object position) 
  and a static rotation matrix (object orientation)
      
*/      
class GateVolumePlacement  : public GateVGlobalPlacement
{
public:
  //! Constructor
  //! The default movement parameters are chosen so that, by default, the volume
  //! is at the origin and the rotation is null (default orientation)
  GateVolumePlacement(GateVVolume* itsObjectInserter,
                      const G4String& itsName,
                      const G4ThreeVector& itsTranslation=G4ThreeVector(),
                      const G4ThreeVector itsRotationAxis=G4ThreeVector(),
                      G4double itsRotationAngle=0.);
    
  //! Destructor
  virtual ~GateVolumePlacement();

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
  //! Get the translation vector
  inline const G4ThreeVector& GetTranslation() 
  { return m_translation;}
  //! Get the rotation axis
  inline const G4ThreeVector& GetRotationAxis() 
  { return m_rotationAxis;}
  //! Get the rotation angle
  inline double GetRotationAngle() 
  { return m_rotationAngle;}

  //! Set the translation vector
  inline void SetTranslation(const G4ThreeVector& val) 
  { m_translation = val; }
  //! Set the rotation axis
  inline void SetRotationAxis(const G4ThreeVector& val) 
  { m_rotationAxis = val; }
  //! Set the rotation angle
  inline void SetRotationAngle(double val) 
  { m_rotationAngle = val; }

  //! Set the phi of the translation vector while keeping mag and theta constant
  inline void SetPhi(G4double val) 
  { m_translation.setPhi(val);  }
  //! Set the theta of the translation vector while keeping mag and phi constant
  inline void SetTheta(G4double val) 
  { m_translation.setTheta(val);  }
  //! Set the mag of the translation vector while keeping theta and phi constant
  inline void SetMag(G4double val) 
  { m_translation.setMag(val);  }
  //@}


  //! \name alignment methods
  //@{
  //! Reset the rotation angle so that the object reverts to its default orientation (aligned along Z)
  inline void AlignToZ() 
  { m_rotationAngle = 0; }
  //! Set the rotation axis and angle so that the object becomes aligned along Y 
  //! (works only if the object was aligned with Z by default)
  inline void AlignToY() 
  { m_rotationAxis = G4ThreeVector(1,0,0) ; 
    m_rotationAngle = -90 * degree; 
  }
  //! Set the rotation axis and angle so that the object becomes aligned along YX
  //! (works only if the object was aligned with Z by default)
  inline void AlignToX() 
  { m_rotationAxis = G4ThreeVector(0,1,0) ; 
    m_rotationAngle = +90 * degree; 
  }
  //@}
      
private:
  //! \name movement parameters
  //@{
  G4ThreeVector m_translation;      	  //!< Translation vector
  G4ThreeVector m_rotationAxis;	  //!< Rotation axis (dimensionless vector)
  G4double m_rotationAngle;      	  //!< Rotation angle
  //@}
    
  //! Messenger
  GateVolumePlacementMessenger* m_Messenger; 

};
//--------------------------------------------------------------------

#endif

