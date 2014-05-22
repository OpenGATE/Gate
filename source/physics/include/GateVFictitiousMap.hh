/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GateVFictitiousMap_hh
#define GateVFictitiousMap_hh

#include "G4ios.hh"
class G4Step;
class G4Material;
class G4Region;
class G4VSolid;
class G4LogicalVolume;
class G4FastTrack;
class G4Box;
#include "G4ThreeVector.hh"
#include <vector>
class GateCrossSectionsTable;
/**
	@author Niklas Rehfeld <rehfeld@imnc.in2p3.fr>
*/

typedef G4Region G4Envelope;


class GateVFictitiousMap
{
public:
  GateVFictitiousMap( G4Envelope* envelope);
  virtual ~GateVFictitiousMap();

  //virtual const G4double operator()(G4double x, G4double y, G4double z, G4double energy)const =0;
  virtual G4double GetCrossSection(const G4ThreeVector& pos, G4double kin_en) const  =0;
  virtual G4double GetMaxCrossSection(G4double kin_en) const =0;
  virtual G4Material* GetMaterial(const G4ThreeVector& pos) const =0;
  virtual void GetMaterials(std::vector<G4Material*>&) const =0;
  // check if everything is correctly initialized, otherwise throw exception	
  virtual void Check() const =0;

 // void RegisterMaxMaterial( G4Material*);
  void RegisterCrossSectionsTable ( const GateCrossSectionsTable*, bool deleteWithThis);
// inline const G4Material* GetMaxMaterial() const;
  inline const G4VSolid* GetSolid() const;
  inline const G4Envelope* GetEnvelope() const;
  inline const G4LogicalVolume* GetLogicalVolume() const;
  inline const G4Box* GetBox() const;

protected:
//  const G4Material* pMaxMaterial;
  const G4VSolid* pSolid;
  const GateCrossSectionsTable* pCrossSectionsTable;
  G4Envelope* pEnvelope;
  const G4LogicalVolume* pLogicalVolume;
  const G4Box* pBox; // NULL if envelope is not G4Box (but treated here, because this will be probably the case in most cases)
  bool m_nDeleteCrossSectionTable;
};

//inline const G4Material* GateVFictitiousMap::GetMaxMaterial() const
 // {return pMaxMaterial;}

inline const G4VSolid* GateVFictitiousMap::GetSolid() const
  { return pSolid;}

inline const G4Envelope* GateVFictitiousMap::GetEnvelope() const
  { return pEnvelope;}
inline const G4LogicalVolume* GateVFictitiousMap::GetLogicalVolume() const
  {return pLogicalVolume;}

inline const G4Box* GateVFictitiousMap::GetBox() const
  {return pBox;}

#endif
