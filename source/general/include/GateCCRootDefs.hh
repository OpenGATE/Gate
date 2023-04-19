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

class GateHit;
class GateDigi;
class GateCCCoincidenceDigi;
class GateComptonCameraCones;

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
    void Fill(GateHit* aHit, std::string layerN);
    GateHit* CreateHit();

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


    //! Returns the energy deposition in G4 units (conversion from MeVs)
    inline G4double GetEnergyIniT() const
    { return energyIniT * MeV;}
    //! Set the energy deposition from a value given in G4 units (conversion into MeVs)
    inline void SetEnergyIniT(G4double aEini)
    { energyIniT = aEini / MeV;}

    //! Returns the energy deposition in G4 units (conversion from MeVs)
    inline G4double GetEnergyFin() const
    { return energyFin * MeV;}
    //! Set the energy deposition from a value given in G4 units (conversion into MeVs)
    inline void SetEnergyFin(G4double aEnergy)
    { energyFin = aEnergy / MeV;}

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


    //! Returns the local position in G4 units (conversion from millimeters)
    inline G4ThreeVector GetsourcePos() const
    { return G4ThreeVector(sPosX,sPosY,sPosZ) * mm ;}
    //! Set the local position from a value given in G4 units (conversion into millimeters)
    inline void SetsourcePos(const G4ThreeVector& sPosition)
    {
        sPosX = sPosition.x() / mm;
        sPosY = sPosition.y() / mm;
        sPosZ = sPosition.z() / mm;
    }
    inline G4double GetSourceEnergy() const
    { return sourceEnergy* MeV;}
    inline void SetSourceEnergy(G4double sEnergy)
    { sourceEnergy = sEnergy / MeV;}

    inline G4int GetSourcePDG() const
    { return sourcePDG;}
    inline void SetSourcePDG(G4int sPDG)
    { sourcePDG = sPDG;}

    inline G4int GetNCrystalConv() const
    { return nCrystalConv;}
    inline void SetNCrystalConv(G4int nConv)
    { nCrystalConv = nConv;}

    inline G4int GetNCrystalCompton() const
    { return nCrystalCompt;}
    inline void SetNCrystalCompton(G4int nCompt)
    { nCrystalCompt = nCompt;}

    inline G4int GetNCrystalRayleigh() const
    { return nCrystalRayl;}
    inline void SetNCrystalRayleigh(G4int nRayl)
    { nCrystalRayl = nRayl;}
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
    Float_t  sPosX, sPosY, sPosZ; 	//!< Local hit position (in millimeters)
    Float_t sourceEnergy;
    Int_t   sourcePDG;
    Int_t   nCrystalConv;
    Int_t   nCrystalCompt;
    Int_t   nCrystalRayl;
    Int_t    eventID; 	      	      	      	//!< Event ID
    Int_t    runID;   	      	      	      	//!< Run ID
    Char_t   processName[40]; 	      	      	//!< Name of the process that generated the hit
    Char_t  layerName[40];
    Int_t    volumeID[ROOT_VOLUMEIDSIZE];     	//!< Volume ID

    Float_t  energyFin;
    Float_t  energyIniT;
    Char_t   postStepProcess[40];

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
    //void Fill(GateDigi* aDigi, int slayerID);
    void Fill(GateDigi* aDigi);
    GateDigi* CreateSingle();

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
    Float_t  energyFin;
    Float_t  energyIni;
    Float_t  globalPosX;
    Float_t  globalPosY;
    Float_t  globalPosZ;
    Float_t  localPosX, localPosY, localPosZ;
    Float_t  sourcePosX;
    Float_t  sourcePosY;
    Float_t  sourcePosZ;
    Float_t  sourceEnergy;
    Int_t    sourcePDG;
    Int_t    nCrystalConv;
    Int_t   nCrystalCompt;
    Int_t   nCrystalRayl;
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

    GateCCCoincidenceDigi* CreateCoincidence();

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
    Float_t  localPosX, localPosY, localPosZ;
    Float_t  sourcePosX;
    Float_t  sourcePosY;
    Float_t  sourcePosZ;
    Float_t  sourceEnergy;
    Int_t    sourcePDG;
    Int_t    nCrystalConv;
    Int_t   nCrystalCompt;
    Int_t   nCrystalRayl;
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


//-----------------------------------------------------------------------------
class GateCCRootConesBuffer
{
public:

    inline GateCCRootConesBuffer() {Clear();}   	      	  //!< Public constructor
    inline virtual ~GateCCRootConesBuffer() {} 	      	  //!< Public destructor

    void Clear();     	      	      	      	  //!< Reset the fields of the structure
    void Fill(GateComptonCameraCones* aCon);
    GateComptonCameraCones* CreateCone();
    //@{
    Float_t  energy1;
    Float_t  energyR;
    Float_t  globalPosX1;
    Float_t  globalPosY1;
    Float_t  globalPosZ1;
    Float_t  globalPosX2;
    Float_t  globalPosY2;
    Float_t  globalPosZ2;

    // Int_t    coincID;
    Int_t    m_nSingles;
    G4bool m_IsTrueCoind;
    //@}


};
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
class GateCCConesTree : public  TTree
{
public:
    inline GateCCConesTree( const G4String& treeName,
                            const G4String& treeDescription="The root tree for cones")
        : TTree(treeName,treeDescription)
    {}
    virtual inline ~GateCCConesTree() {}

    void Init(GateCCRootConesBuffer& buffer);
    static void SetBranchAddresses(TTree* ,GateCCRootConesBuffer& buffer);
};
//-----------------------------------------------------------------------------

#endif
