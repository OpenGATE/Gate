/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateCCRootHitBuffer
  \class  GateCCHitTree
  \class  GateCCRootSingleBuffer
  \class  GateCCSingleTree
*/

#ifndef GateCCRootHitBuffer_H
#define GateCCRootHitBuffer_H

#include "globals.hh"
#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>
#include <G4PhysicalConstants.hh>

#include <TROOT.h>
#include <TTree.h>

class GateCrystalHit;
class GateSingleDigi;
class GateCCCoincidenceDigi;

#define ROOT_VOLUMEIDSIZE 10
//-----------------------------------------------------------------------------
/*! \class  GateCCRootHitBuffer
  \brief  ROOT structure to store hits f
*/
class GateCCRootHitBuffer
{
public:

  inline GateCCRootHitBuffer() {Clear();}   	      	//!< Public constructor
  inline virtual ~GateCCRootHitBuffer() {} 	      	  //!< Public destructor

  void Clear();     	      	      	      	  //!< Reset the fields of the structure
  void Fill(GateCrystalHit* aHit, std::string layerN);
  GateCrystalHit* CreateHit();

  //! Returns the time in G4 units (conversion from seconds)
  inline G4double GetTime() const
  { return time * second;}
  //! Set the time from a value expressed in G4 units (conversion into seconds)
  inline void SetTime(G4double aTime)
  { time = aTime / second;}

  //! Returns the time in G4 units (conversion from seconds)
  inline G4double GetTrackLocalTime() const
  { return trackLocalTime * second;}
  //! Set the time from a value expressed in G4 units (conversion into seconds)
  inline void SetTrackLocalTime(G4double aTime)
  { trackLocalTime = aTime / second;}

  //! Returns the energy deposition in G4 units (conversion from MeVs)
  inline G4double GetEdep() const
  { return edep * MeV;}
  //! Set the energy deposition from a value given in G4 units (conversion into MeVs)
  inline void SetEdep(G4double anEnergy)
  { edep = anEnergy / MeV;}

  //! Returns the step length in G4 units (conversion from millimeters)
  inline G4double GetStepLength() const
  { return stepLength * mm;}
  //! Set the step length from a value given in G4 units (conversion into millimeters)
  inline void SetStepLength(G4double aLength)
  { stepLength = aLength / mm;}

  //! Returns the track length in G4 units (conversion from millimeters)
  inline G4double GetTrackLength() const
  { return trackLength * mm;}
  //! Set the track length from a value given in G4 units (conversion into millimeters)
  inline void SetTrackLength(G4double aLength)
  { trackLength = aLength / mm;}

  //! Returns the global position in G4 units (conversion from millimeters)
  inline G4ThreeVector GetPos() const
  { return G4ThreeVector(posX,posY,posZ) * mm ;}
  //! Set the global position from a value given in G4 units (conversion into millimeters)
  inline void SetPos(const G4ThreeVector& aPosition)
  {
    posX = aPosition.x() / mm;
    posY = aPosition.y() / mm;
    posZ = aPosition.z() / mm;
  }

  //! Returns the local position in G4 units (conversion from millimeters)
  inline G4ThreeVector GetLocalPos() const
  { return G4ThreeVector(localPosX,localPosY,localPosZ) * mm ;}
  //! Set the local position from a value given in G4 units (conversion into millimeters)
  inline void SetLocalPos(const G4ThreeVector& aPosition)
  {
    localPosX = aPosition.x() / mm;
    localPosY = aPosition.y() / mm;
    localPosZ = aPosition.z() / mm;
  }

  //@}

  //! \name Data fields
  //@{
  Int_t    PDGEncoding;     	      	      	//!< PDG encoding of the particle
  Int_t    trackID; 	      	      	      	//!< Track ID
  Int_t    parentID;	      	      	      	//!< Parent ID
  Double_t time;    	      	      	      	//!< Time of the hit (in seconds)
  Double_t trackLocalTime;    	      	      //!< Time of the current track (in seconds)
  Float_t  edep;    	      	      	      	//!< Deposited energy (in MeVs)
  Float_t  stepLength;      	      	      	//!< Step length (in millimeters)
  Float_t  trackLength;      	      	      	//!< Track length (in millimeters)
  Float_t  posX,posY,posZ;  	      	      	//!< Global hit position (in millimeters)
  Float_t  localPosX, localPosY, localPosZ; 	//!< Local hit position (in millimeters)
  Int_t    eventID; 	      	      	      	//!< Event ID
  Int_t    runID;   	      	      	      	//!< Run ID
  Char_t   processName[40]; 	      	      	//!< Name of the process that generated the hit
  Char_t  layerName[40];
  Int_t    volumeID[ROOT_VOLUMEIDSIZE];     	//!< Volume ID
  //@}

};
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
class GateCCHitTree : public  TTree
{
public:
  inline GateCCHitTree( const G4String& treeName,
                        const G4String& treeDescription="The root tree for hits")
    : TTree(treeName,treeDescription)
  {}
  virtual inline ~GateCCHitTree() {}

  void Init(GateCCRootHitBuffer& buffer);
  static void SetBranchAddresses(TTree* hitTree,GateCCRootHitBuffer& buffer);
};
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
class GateCCRootSingleBuffer
{
public:

  inline GateCCRootSingleBuffer() {Clear();}   	      	  //!< Public constructor
  inline virtual ~GateCCRootSingleBuffer() {} 	      	  //!< Public destructor

  void Clear();     	      	      	      	  //!< Reset the fields of the structure
  //void Fill(GateSingleDigi* aDigi, int slayerID);
  void Fill(GateSingleDigi* aDigi);
  GateSingleDigi* CreateSingle();

  //! Returns the time in G4 units (conversion from seconds)
  inline G4double GetTime() const
    { return time * second;}
  //! Set the time from a value expressed in G4 units (conversion into seconds)
  inline void SetTime(G4double aTime)
    { time = aTime / second;}


  //! \name Data fields
  //@{
  Int_t    runID;
  Int_t    eventID;
  Double_t time;
  Float_t  energy;
  Float_t  globalPosX;
  Float_t  globalPosY;
  Float_t  globalPosZ;
  //Int_t    layerID;
  Char_t   layerName[40];
  Int_t    sublayerID;
   Int_t    volumeID[ROOT_VOLUMEIDSIZE];     	//!< Volume ID
  //@}
};
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
class GateCCSingleTree : public  TTree
{
public:
  inline GateCCSingleTree( const G4String& treeName,
                           const G4String& treeDescription="The root tree for singles")
    : TTree(treeName,treeDescription)
  {}
  virtual inline ~GateCCSingleTree() {}

  void Init(GateCCRootSingleBuffer& buffer);
  static void SetBranchAddresses(TTree* singleTree,GateCCRootSingleBuffer& buffer);
};
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class GateCCRootCoincBuffer
{
public:

  inline GateCCRootCoincBuffer() {Clear();}   	      	  //!< Public constructor
  inline virtual ~GateCCRootCoincBuffer() {} 	      	  //!< Public destructor

  void Clear();     	      	      	      	  //!< Reset the fields of the structure
  void Fill(GateCCCoincidenceDigi* aDigi);



  //! Returns the time in G4 units (conversion from seconds)
  inline G4double GetTime() const
    { return time * second;}
  //! Set the time from a value expressed in G4 units (conversion into seconds)
  inline void SetTime(G4double aTime)
    { time = aTime / second;}

  //! \name Data fields
  //@{
  Int_t    coincID;
  Int_t    runID;
  Int_t    eventID;
  Double_t time;
  Float_t  energy;
  Float_t  energyFin;
  Float_t  energyIni;
  Float_t  globalPosX;
  Float_t  globalPosY;
  Float_t  globalPosZ;
  //Int_t    layerID;
  Char_t   layerName[40];
  Int_t    sublayerID;
   Int_t    volumeID[ROOT_VOLUMEIDSIZE];     	//!< Volume ID
  //@}
};
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
class GateCCCoincTree : public  TTree
{
public:
  inline GateCCCoincTree( const G4String& treeName,
                           const G4String& treeDescription="The root tree for coincidences ")
    : TTree(treeName,treeDescription)
  {}
  virtual inline ~GateCCCoincTree() {}

  void Init(GateCCRootCoincBuffer& buffer);
  static void SetBranchAddresses(TTree* ,GateCCRootCoincBuffer& buffer);
};
//-----------------------------------------------------------------------------

#endif
