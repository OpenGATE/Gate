/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GateTessellated_h
#define GateTessellated_h 1

#include "GateVVolume.hh"
#include "GateVolumeManager.hh"

class G4TessellatedSolid;
class G4LogicalVolume;
class G4VPhysicalVolume;

class GateTessellatedMessenger;

class GateTessellated : public GateVVolume
{
public:
  GateTessellated( G4String const& itsName, G4bool acceptsChildren = true, G4int depth = 0 );
  GateTessellated( G4String const& itsName, G4String const& itsMaterialName );

  virtual ~GateTessellated();

  FCT_FOR_AUTO_CREATOR_VOLUME(GateTessellated)

  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume( G4Material*, G4bool );
  virtual void DestroyOwnSolidAndLogicalVolume();

  // Declaration of pure virtual method
  inline virtual G4double GetHalfDimension( size_t ) { return 0.; }

  void SetPathToSTLFile( G4String );

private:
  void ReadSTL_ASCII();
  void ReadSTL_Binary();
  void DescribeMyself(size_t);
  G4double ComputeMyOwnVolume() const;

private:
  G4TessellatedSolid*     m_tessellated_solid;
  G4LogicalVolume*        m_tessellated_log;
  // G4VPhysicalVolume*      m_tessellated_phys; not used 

private:
  G4String m_PathToSTLFile;
  GateTessellatedMessenger* m_Messenger;
  G4String FacetType;
  unsigned long nbFacets;
};

MAKE_AUTO_CREATOR_VOLUME(tessellated,GateTessellated)

#endif
