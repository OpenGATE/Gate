/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GateCone_h
#define GateCone_h 1

#include "globals.hh"

#include "GateVVolume.hh"
#include "GateVolumeManager.hh"

#include "G4Cons.hh"
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4Material;


class GateConeMessenger;

/*! \class  GateCone
  \brief  The GateCone creates a solid and a logical volume for a cone

  - GateCone - by Daniel.Strul@iphe.unil.ch

*/
class GateCone  : public GateVVolume
{
public:

  GateCone(const G4String& itsName,
           G4bool acceptsChildren=true,
           G4int depth=0);

  //! Constructor
  GateCone(const G4String& itsName,const G4String& itsMaterialName,
           G4double itsRmax1, G4double itsRmax2,
           G4double itsHeight,
           G4double itsRmin1=0.,G4double itsRmin2=0.,
           G4double itsSPhi=0., G4double itsDPhi=2*M_PI,
           G4bool acceptsChildren=true,
           G4int depth=0);
  //! Destructor
  virtual ~GateCone();

  FCT_FOR_AUTO_CREATOR_VOLUME(GateCone)

  //! \name Implementations of pure virtual methods declared by the base-class
  //@{

  virtual G4LogicalVolume*   ConstructOwnSolidAndLogicalVolume(G4Material*, G4bool);
  //     virtual G4VPhysicalVolume* ConstructOwnPhysicalVolume();

  //! Implementation of the pure virtual method DestroyOwnSolidAndVolume() declared by the base-class.
  //! Destroy the solid and logical volume created by ConstructOwnSolidAndLogical()
  virtual void DestroyOwnSolidAndLogicalVolume();

  /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
    \brief a description of the creator

    \param indent: the print-out indentation (cosmetic parameter)
  */
  virtual void DescribeMyself(size_t indent);

  //! Implementation of the pure virtual method GetHalfDimension() declared by the base-class
  //! Must return the half-size of the cone along an axis (X=0, Y=1, Z=2)
  //! Accurate only for full cones
  inline G4double GetHalfDimension(size_t axis)
  { if (axis==2)
      return GetConeHalfHeight();
    else
      return .5* ( m_coneRmax1 + m_coneRmax2 );
  }

  //@}

  //! \name getters and setters
  //@{

  //! Get the height
  inline G4double GetConeHeight()          {return m_coneHeight;};
  //! Get the half of the height
  inline G4double GetConeHalfHeight()      {return m_coneHeight/2.;};
  //! Get the internal diameter at end 1
  inline G4double GetConeRmin1()     {return m_coneRmin1;};
  //! Get the external diameter at end 1
  inline G4double GetConeRmax1()     {return m_coneRmax1;};
  //! Get the internal diameter at end 2
  inline G4double GetConeRmin2()     {return m_coneRmin2;};
  //! Get the external diameter at end 2
  inline G4double GetConeRmax2()     {return m_coneRmax2;};
  //! Get the start phi angle
  inline G4double GetConeSPhi()     {return m_coneSPhi;};
  //! Get the angular span for the phi angle
  inline G4double GetConeDPhi()     {return m_coneDPhi;};

  //! Set the height
  void SetConeHeight   (G4double val)
  {  m_coneHeight = val; /*ComputeParameters();*/ }
  //! Set the internal diameter at end 1
  void SetConeRmin1  (G4double val)
  {  m_coneRmin1 = val; /*ComputeParameters();*/ }
  //! Set the external diameter at end 1
  void SetConeRmax1  (G4double val)
  { m_coneRmax1 = val; /*ComputeParameters();*/ }
  //! Set the internal diameter at end 2
  void SetConeRmin2  (G4double val)
  { m_coneRmin2 = val; /*ComputeParameters();*/ }
  //! Set the external diameter at end 2
  void SetConeRmax2  (G4double val)
  { m_coneRmax2 = val; /*ComputeParameters();*/ }
  //! Set the start phi angle
  void SetConeSPhi  (G4double val)
  {  m_coneSPhi = val; /*ComputeParameters();*/ }
  //! Set the angular span for the phi angle
  void SetConeDPhi  (G4double val)
  {  m_coneDPhi = val; /*ComputeParameters(); */}

  //@}

private:
  //! \name own geometry
  //@{
  G4Cons*          m_cone_solid;       	    //!< Solid pointer
  G4LogicalVolume*   m_cone_log;	      	    //!< logical volume pointer
  //@}

  //! \name parameters
  //@{
  G4double m_coneHeight;   	      	      	    //!< height
  G4double m_coneRmin1;   	      	      	    //!< internal diameter at end 1
  G4double m_coneRmax1;  	      	      	    //!< external diameter at end 1
  G4double m_coneRmin2;  	      	      	    //!< internal diameter at end 2
  G4double m_coneRmax2;  	      	      	    //!< external diameter at end 2
  G4double m_coneSPhi;   	      	      	    //!< start phi angle
  G4double m_coneDPhi;   	      	      	    //!< angular span for the phi angle
  //@}

  //! Messenger
  GateConeMessenger* m_Messenger;

};

MAKE_AUTO_CREATOR_VOLUME(cone,GateCone)

#endif
