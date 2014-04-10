/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateSphere_h
#define GateSphere_h 1

#include "globals.hh"
#include "G4Sphere.hh"

#include "GateVVolume.hh"
#include "GateVolumeManager.hh"

class G4VisAttributes;

class GateSphereMessenger;

/*! \class  GateSphere
  \brief  The GateSphere creates a solid and a logical volume for a sphere

  - GateSphere - by Daniel.Strul@iphe.unil.ch

*/
class GateSphere  : public GateVVolume
{
public:
  //! Constructor

  GateSphere(const G4String& itsName,
             G4bool acceptsChildren=true,
             G4int depth=0);


  GateSphere(const G4String& itsName, const G4String& itsMaterialName,
             G4double itsRmax,
             G4double itsRmin=0.,
             G4double itsSPhi=0., G4double itsDPhi=2*M_PI,
             G4double itsSTheta=0., G4double itsDTheta=M_PI,
             G4bool acceptsChildren=true,
             G4int depth=0);
  //! Destructor
  virtual ~GateSphere();

  FCT_FOR_AUTO_CREATOR_VOLUME(GateSphere)

  //! \name Implementations of pure virtual methods declared by the base-class
  //@{

  //! Implementation of the pure virtual method ConstructOwnSolidAndLogical() declared by the base-class.
  //! Construct a new sphere-solid and its logical volume.
  //! If flagUpdateOnly is set to 1, the sphere is updated rather than rebuilt.
  virtual G4LogicalVolume*   ConstructOwnSolidAndLogicalVolume(G4Material*, G4bool);
  //     virtual G4VPhysicalVolume* ConstructOwnPhysicalVolume();

  //! Implementation of the pure virtual method DestroyOwnSolidAndVolume() declared by the base-class.
  //! Destroy the solid and logical volume created by ConstructOwnSolidAndLogical()
  virtual void DestroyOwnSolidAndLogicalVolume();

  /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
    \brief a description of the creator

    \param indent: the print-out indentation (cosmetic parameter)
  */
  virtual void DescribeMyself(size_t level);

  //! Implementation of the pure virtual method GetHalfDimension() declared by the base-class
  //! Must return the half-size of the sphere along an axis (X=0, Y=1, Z=2)
  //! Returns the radius: accurate only for full spheres
  inline G4double GetHalfDimension(size_t )
  {return m_sphereRmax;};

  //! Overload of the dummy virtual method ComputeMyOwnVolume() defined by the base-class
  //! Returns the volume of the solid
  G4double ComputeMyOwnVolume()  const;

  //@}

  //! \name getters and setters
  //@{

  //! Get the internal diameter
  inline G4double GetSphereRmin()     {return m_sphereRmin;};
  //! Get the external diameter
  inline G4double GetSphereRmax()     {return m_sphereRmax;};
  //! Get the start phi angle
  inline G4double GetSphereSPhi()     {return m_sphereSPhi;};
  //! Get the angular span for the phi angle
  inline G4double GetSphereDPhi()     {return m_sphereDPhi;};
  //! Get the start theta angle
  inline G4double GetSphereSTheta()     {return m_sphereSTheta;};
  //! Get the angular span for the theta angle
  inline G4double GetSphereDTheta()     {return m_sphereDTheta;};

  //! Set the internal diameter
  void SetSphereRmin  (G4double val)
  {  m_sphereRmin = val; /*ComputeParameters();*/ }
  //! Set the external diameter
  void SetSphereRmax  (G4double val)
  {  m_sphereRmax = val; /*ComputeParameters();*/ }
  //! Set the start phi angle
  void SetSphereSPhi  (G4double val)
  {  m_sphereSPhi = val; /*ComputeParameters();*/ }
  //! Set the angular span for the phi angle
  void SetSphereDPhi  (G4double val)
  {  m_sphereDPhi = val; /*ComputeParameters();*/ }
  //! Set the start theta angle
  void SetSphereSTheta  (G4double val)
  { m_sphereSTheta = val; /*ComputeParameters();*/ }
  //! Set the angular span for the theta angle
  void SetSphereDTheta  (G4double val)
  {  m_sphereDTheta = val; /*ComputeParameters();*/ }

  //@}

private:
  //! \name own geometry
  //@{
  G4Sphere*          m_sphere_solid;       	    //!< Solid pointer
  G4LogicalVolume*   m_sphere_log; 	      	    //!< logical volume pointer
  //@}

  //! \name parameters
  //@{
  G4double m_sphereRmin;   	      	      	    //!< internal diameter
  G4double m_sphereRmax;   	      	      	    //!< external diameter
  G4double m_sphereSPhi;   	      	      	    //!< start phi angle
  G4double m_sphereDPhi;   	      	      	    //!< angular span for the phi angle
  G4double m_sphereSTheta;   	      	    //!< start theta angle
  G4double m_sphereDTheta;   	      	    //!< angular span for the theta angle
  //@}

  //! Object visualisation attribute object.
  //! It is passed to the logical volume each time the logical volume is created
  G4VisAttributes *m_own_visAtt;

  //! Messenger
  GateSphereMessenger* m_Messenger;

};

MAKE_AUTO_CREATOR_VOLUME(sphere,GateSphere)

#endif
