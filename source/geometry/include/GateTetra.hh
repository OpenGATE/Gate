/*
 * GateTetra.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATETETRA_H
#define GATETETRA_H

#include "globals.hh"
#include "GateVVolume.hh"
#include "GateVolumeManager.hh"
#include "G4ThreeVector.hh"
#include "G4VisExtent.hh"
#include "G4Tet.hh"

class G4Tet;
class GateTetraMessenger;

class GateTetra : public GateVVolume
{
public:
  //! Constructor
  GateTetra(const G4String& itsName,
                 G4bool acceptsChildren=true,
                 G4int depth=0);

  GateTetra(const G4String& itsName,
                 const G4String& itsMaterialName,
                 G4ThreeVector itsP1,
                 G4ThreeVector itsP2,
                 G4ThreeVector itsP3,
                 G4ThreeVector itsP4,
                 G4bool acceptsChildren=true,
                 G4int depth=0);
  //! Destructor
  virtual ~GateTetra();

  FCT_FOR_AUTO_CREATOR_VOLUME(GateTetra)

  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly);

  virtual void DestroyOwnSolidAndLogicalVolume();

  /*! \brief Implementation of the virtual method DescribeMyself(), to print-out
    \brief a description of the creator

    \param indent: the print-out indentation (cosmetic parameter)
  */
  virtual void DescribeMyself(size_t level);

  inline G4double GetHalfDimension(size_t axis) {
    G4VisExtent extent = m_tetra_solid->GetExtent();
    if (axis==0) {
	return 0.5*(extent.GetXmax() - extent.GetXmin());
    } else if (axis==1) {
	return 0.5*(extent.GetYmax() - extent.GetYmin());
    } else if (axis==2) {
	return 0.5*(extent.GetZmax() - extent.GetZmin());
    }
    return 0.0;
  }

  //! Overload of the dummy virtual method ComputeMyOwnVolume() defined by the base-class
  //! Returns the volume of the solid
  G4double ComputeMyOwnVolume()  const;

  //! \name getters and setters
  //@{
  inline G4ThreeVector GetTetraP1() 	{ return m_p1; }
  inline G4ThreeVector GetTetraP2() 	{ return m_p2; }
  inline G4ThreeVector GetTetraP3() 	{ return m_p3; }
  inline G4ThreeVector GetTetraP4() 	{ return m_p4; }

  void SetTetraP1(G4ThreeVector val)	{ m_p1 = val; }
  void SetTetraP2(G4ThreeVector val)	{ m_p2 = val; }
  void SetTetraP3(G4ThreeVector val)	{ m_p3 = val; }
  void SetTetraP4(G4ThreeVector val)	{ m_p4 = val; }

  //@}


private:
  //! \name own geometry
  //@{
  G4Tet*          m_tetra_solid;       	    //!< Solid pointer
  G4LogicalVolume*   m_tetra_log; 	      	    //!< logical volume pointer
  //@}

  //! \name parameters
  //@{
  G4ThreeVector m_p1;		//!< vertex 1
  G4ThreeVector m_p2;		//!< vertex 2
  G4ThreeVector m_p3;		//!< vertex 3
  G4ThreeVector m_p4;		//!< vertex 4
  //@}

  //! Messenger
  GateTetraMessenger* m_Messenger;
};

MAKE_AUTO_CREATOR_VOLUME(tetra,GateTetra)


#endif /* GATETETRA_H */
