/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*  Update for Optical Photons: V. Cuplov   15 Feb. 2012
            - Added ROOT structure to store phantom hits
*/

#ifndef GateRootHitBuffer_H
#define GateRootHitBuffer_H

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "globals.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"

#include "TROOT.h"
#include "TTree.h"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

class GateHit;
class GateDigi;

class GateCoincidenceDigi;

#define ROOT_VOLUMEIDSIZE 10
#define ROOT_OUTPUTIDSIZE 6

/*! \namespace  GateRootDefs
    \brief  Namespace to provide general-purpose methods to the ROOT-based classes
    
    - GateTools - by Daniel.Strul@iphe.unil.ch
    
    - The GateRootDefs namespace is a collection of data and functions providing 
      some general-purpose definitions to the ROOT-based classes:
      - Methods allowing to change the output-ID names
*/      

namespace GateRootDefs
{
  void SetDefaultOutputIDNames();
  void SetOutputIDName(char * anOutputIDName, size_t depth);
  G4bool GetRecordSeptalFlag(); // HDS : record septal penetration
}

/*! \class  GateRootHitBuffer
    \brief  ROOT structure to store hits for GateToRoot and GateHitFileReader
    
    - GateRootHitBuffer - by Giovanni.Santin@cern.ch (May 1, 2002)
    
    - This structure was initally declared in GateToRoot. To insure consistency between
      GateToRoot and GatHitFileReader, I have made it a separate class.
*/      
class GateRootHitBuffer
{
  public:

    inline GateRootHitBuffer() { Clear();}   	      	  //!< Public constructor
    inline virtual ~GateRootHitBuffer() {} 	      	  //!< Public destructor

    void Clear();     	      	      	      	  //!< Reset the fields of the structure
    void Fill(GateHit* aHit);
    GateHit* CreateHit();

    //! \name getters and setters for unit-dependent fields
    //@{
    inline void SetCCFlag (G4bool val) {m_CCflag=val;}
    inline G4bool GetCCFlag () {return m_CCflag;}


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

    //! Returns the source position in G4 units (conversion from millimeters)
    inline G4ThreeVector GetSourcePos() const
      { return G4ThreeVector(sourcePosX,sourcePosY,sourcePosZ) * mm ;}
    //! Set the source position from a value given in G4 units (conversion into millimeters)
    inline void SetSourcePos(const G4ThreeVector& aPosition)
      { 
      	sourcePosX = aPosition.x() / mm; 
	sourcePosY = aPosition.y() / mm; 
	sourcePosZ = aPosition.z() / mm; 
      }

    //! Returns the scanner axial position in G4 units (conversion from millimeters)
    inline G4double GetAxialPos() const
      { return axialPos * mm;}
    //! Set the scanner axial position from a value given in G4 units (conversion into millimeters)
    inline void SetAxialPos(G4double anAxialPos)
      { axialPos = anAxialPos / mm;}

    //! Returns the scanner rotation angle in G4 units (conversion from degrees)
    inline G4double GetRotationAngle() const
      { return rotationAngle * degree;}
    //! Set the scanner rotation angle from a value given in G4 units (conversion into degrees)
    inline void SetRotationAngle(G4double anAngle)
      { rotationAngle = anAngle / degree;}

    //OK GND for CC
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


    //@}
    //! \name Data fields
    //@{
    Int_t    PDGEncoding;     	      	      	//!< PDG encoding of the particle
    Int_t    trackID; 	      	      	      	//!< Track ID
    Int_t    parentID;	      	      	      	//!< Parent ID
    Double_t time;    	      	      	      	//!< Time of the hit (in seconds)
    Double_t trackLocalTime;    	      	      	//!< Time of the current track (in seconds)
    Float_t  edep;    	      	      	      	//!< Deposited energy (in MeVs)
    Float_t  stepLength;      	      	      	//!< Step length (in millimeters)
    Float_t  trackLength;      	      	      	//!< Track length (in millimeters)
    Float_t  posX,posY,posZ;  	      	      	//!< Global hit position (in millimeters)
    Float_t  momDirX,momDirY,momDirZ;              //!< Global hit momentum
    Float_t  localPosX, localPosY, localPosZ; 	//!< Local hit position (in millimeters)
    Int_t    outputID[ROOT_OUTPUTIDSIZE];	//!< 6-position output ID
    Int_t    photonID;	      	      	      	//!< Photon ID
    Int_t    nPhantomCompton; 	      	      	//!< Number of Compton interactions in the phantom
    Int_t    nCrystalCompton; 	      	      	//!< Number of Compton interactions in the crystam
    Int_t    nPhantomRayleigh; 	      	      	//!< Number of Rayleigh interactions in the phantom
    Int_t    nCrystalRayleigh; 	      	      	//!< Number of Rayleigh interactions in the crystam
    Int_t    primaryID;       	      	      	//!< Primary ID
    Float_t  sourcePosX,sourcePosY,sourcePosZ;	//!< Global decay position (in millimeters)
    Int_t    sourceID;	      	      	      	//!< Source ID
    Int_t    eventID; 	      	      	      	//!< Event ID
    Int_t    runID;   	      	      	      	//!< Run ID
    Float_t  axialPos;	      	      	      	//!< Scanner axial position (in millimeters)
    Float_t  rotationAngle;           	      	//!< Rotation angle (in degrees)
    Char_t   processName[40]; 	      	      	//!< Name of the process that generated the hit
    Char_t   comptonVolumeName[40];   	      	//!< Name of the last phantom-volume generating a Compton
    Char_t   RayleighVolumeName[40];   	      	//!< Name of the last phantom-volume generating a Rayleigh
    Int_t    volumeID[ROOT_VOLUMEIDSIZE];     	//!< Volume ID
    Int_t    septalNb;							//!< HDS : septal penetration
    Int_t sourceType = 0; //Type of gamma source (check ExtendedVSource)
    Int_t decayType = 0; //Type of positronium decay (check ExtendedVSource)
    Int_t gammaType = 0; //Gamma type - single, annhilation, prompt (check ExtendedVSource)

    //OK GND for CC
    G4bool m_CCflag;
    Float_t sourceEnergy;
    Int_t   sourcePDG;
    Int_t   nCrystalConv;
    Int_t   nCrystalCompt;
    Int_t   nCrystalRayl;
    Float_t  energyFin;
    Float_t  energyIniT;
    Char_t   postStepProcess[40];
    //@}



};


/*! \class  GateHitTree
    \brief  ROOT tree to store hits
    
    - GateHitTree - by Giovanni.Santin@cern.ch (May 1, 2002)
    
    - This tree, initally declared in GateToRoot, was changed 
      into a separate class.
*/      
class GateHitTree : public  TTree
{
  public:
    inline GateHitTree( const G4String& treeName,
    			const G4String& treeDescription="The root tree for hits")
      : TTree(treeName,treeDescription)
      {}
    virtual inline ~GateHitTree() {}

    void Init(GateRootHitBuffer& buffer);
    static void SetBranchAddresses(TTree* hitTree,GateRootHitBuffer& buffer);

};


/*! \class  GateRootSingleBuffer
    \brief  ROOT structure to store singles for GateToRoot
    
    - GateRootHitBuffer - by Giovanni.Santin@cern.ch (May 1, 2002)
    
    - This structure, initally declared in GateToRoot, was changed 
      into a separate class.
*/      
class GateRootSingleBuffer
{
  public:

    inline GateRootSingleBuffer() {Clear();}   	      	  //!< Public constructor
    inline virtual ~GateRootSingleBuffer() {} 	      	  //!< Public destructor

    void Clear();     	      	      	      	  //!< Reset the fields of the structure
    void Fill(GateDigi* aDigi);



    inline void SetCCFlag (G4bool val) {m_CCflag=val;}
    inline G4bool GetCCFlag () {return m_CCflag;}
    G4bool m_CCflag;

    //! \name Data fields
    //@{

    Int_t    runID;
    Int_t    eventID;
    Int_t    sourceID;
    Float_t  sourcePosX;
    Float_t  sourcePosY;
    Float_t  sourcePosZ;
    Double_t time;
    Float_t  energy;
    Float_t  globalPosX;
    Float_t  globalPosY;
    Float_t  globalPosZ;
    Int_t    outputID[ROOT_OUTPUTIDSIZE];
    Int_t    volumeID[ROOT_VOLUMEIDSIZE];
    Int_t    comptonPhantom; 
    Int_t    comptonCrystal;    
    Int_t    RayleighPhantom; 
    Int_t    RayleighCrystal;    
    Float_t  axialPos;
    Float_t  rotationAngle;    
    Char_t   comptonVolumeName[40];
    Char_t   RayleighVolumeName[40];

    Float_t  localPosX, localPosY, localPosZ;
    Float_t  sourceEnergy;
    Int_t    sourcePDG;
    Int_t    nCrystalConv;
    Int_t   nCrystalCompt;
    Int_t   nCrystalRayl;
    Float_t  energyFin;
    Float_t  energyIni;

    Int_t    septalNb;							//!< HDS : septal penetration
    //@}
};

/*! \class  GateSingleTree
    \brief  ROOT tree to store singles
    
    - GateSingleTree - by Giovanni.Santin@cern.ch (May 1, 2002)
    
    - This tree, initally declared in GateToRoot, was changed 
      into a separate class.
*/      
class GateSingleTree : public  TTree
{
  public:
    inline GateSingleTree( const G4String& treeName,
    			const G4String& treeDescription="The root tree for singles")
      : TTree(treeName,treeDescription)
      {}
    virtual inline ~GateSingleTree() {}

    void Init(GateRootSingleBuffer& buffer);


};


/*! \class  GateRootCoincBuffer
    \brief  ROOT structure to store coincidences for GateToRoot
    
    - GateRootHitBuffer - by Giovanni.Santin@cern.ch (May 1, 2002)
    
    - This structure, initally declared in GateToRoot, was changed 
      into a separate class.
*/      
class GateRootCoincBuffer
{
  public:

    inline GateRootCoincBuffer() {Clear();}   	  //!< Public constructor
    inline virtual ~GateRootCoincBuffer() {} 	      	  //!< Public destructor

    void Clear();     	      	      	      	  //!< Reset the fields of the structure
    void Fill(GateCoincidenceDigi* aDigi);

    G4double ComputeSinogramTheta();
    G4double ComputeSinogramS();



    inline void SetCCFlag (G4bool val) {m_CCflag=val;}
    inline G4bool GetCCFlag() {return m_CCflag;}
    G4bool m_CCflag;

    //! \name Data fields
    //@{

    Int_t    runID;
    Float_t  axialPos;
    Float_t  rotationAngle;    

    Int_t    eventID1;
    Int_t    sourceID1;
    Float_t  sourcePosX1;
    Float_t  sourcePosY1;
    Float_t  sourcePosZ1;
    Double_t time1;
    Float_t  energy1;
    Float_t  globalPosX1;
    Float_t  globalPosY1;
    Float_t  globalPosZ1;
    Int_t    outputID1[ROOT_OUTPUTIDSIZE];
    Int_t    comptonPhantom1;
    Int_t    comptonCrystal1;   
    Int_t    RayleighPhantom1;
    Int_t    RayleighCrystal1;   
    Char_t   comptonVolumeName1[40];
    Char_t   RayleighVolumeName1[40];

    Int_t    eventID2;
    Int_t    sourceID2;
    Float_t  sourcePosX2;
    Float_t  sourcePosY2;
    Float_t  sourcePosZ2;
    Double_t time2;
    Float_t  energy2;
    Float_t  globalPosX2;
    Float_t  globalPosY2;
    Float_t  globalPosZ2;
    Int_t    outputID2[ROOT_OUTPUTIDSIZE];
    Int_t    comptonPhantom2;
    Int_t    comptonCrystal2;    
    Int_t    RayleighPhantom2;
    Int_t    RayleighCrystal2;    
    Char_t   comptonVolumeName2[40];
    Char_t   RayleighVolumeName2[40];

    Float_t  sinogramTheta;
    Float_t  sinogramS;
    //@}

};


/*! \class  GateCoincTree
    \brief  ROOT tree to store singles
    
    - GateCoincTree - by Giovanni.Santin@cern.ch (May 1, 2002)
    
    - This tree, initally declared in GateToRoot, was changed 
      into a separate class.
*/      
class GateCoincTree : public  TTree
{
  public:
    inline GateCoincTree( const G4String& treeName,
    			  const G4String& treeDescription="The root tree for coincidences")
      : TTree(treeName,treeDescription)
      {}
    virtual inline ~GateCoincTree() {}

    void Init(GateRootCoincBuffer& buffer);
};



#endif
#endif
