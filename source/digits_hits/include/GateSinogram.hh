/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*----------------------
   Modifications history

     Gate 6.2

	C. Comtat, CEA/SHFJ, 10/02/2011	   Allows for virtual crystals, needed to simulate ecat like sinogram output for Biograph scanners

----------------------*/

#ifndef GateSinogram_H
#define GateSinogram_H

#include "GateConfiguration.h"
#include "globals.hh"
#include <fstream>

/*! \class  GateSinogram
    \brief  Structure to store the sinogram sets from a PET simulation

    - GateSinogram - by By jan@shfj.cea.fr & comtat@shfj.cea.fr (december 2002)

    - This structure is generated during a PET simulation by GateToSinogram. It can be stored
      into an output file using a set-writer such as GateSinoToEcat7

    \sa GateToSinogram, GateSinoToEcat7
*/
class GateSinogram
{
  public:
    typedef unsigned short SinogramDataType;

  public:

    inline GateSinogram();       	      	      	  //!< Public constructor
    inline virtual ~GateSinogram() {Reset();}    	  //!< Public destructor

    //! Reset the sinogrames and prepare a new acquisition
    void Reset(size_t ringNumber=0, size_t crystalNumber=0, size_t radialElemNb=0, size_t virtualRingNumber=0, size_t virtualCrystalPerBlockNumber=0);

    //! Clear the matrix and prepare a new run
    void ClearData(size_t frameID, size_t gateID, size_t dataID, size_t bedID);

    //! Apply spatial blurring to hited crystal
    void CrystalBlurring( G4int *ringID, G4int *crystalID, G4double ringResolution,
                          G4double crystalResolution );

    //! Store a digi into a projection
    G4int Fill( G4int ring1ID, G4int ring2ID, G4int crystal1ID, G4int crystal2ID, int signe);

    //! Store a digi into randoms array
    G4int FillRandoms( G4int ring1ID, G4int ring2ID);

    //! Returns the 2D sino ID for a given pair of rings
    G4int GetSinoID( G4int ring1ID, G4int ring2ID);

    //! \name getters and setters
    //@{

    //! Returns the number of crystal rings
    inline size_t GetRingNb() const
      { return m_ringNb;}
    //! Set the number of rings
    inline void SetRingNb(size_t aNb)
      { m_ringNb = aNb;}

    // C. Comtat, February 2011: Required to simulate Biograph output sinograms with virtual crystals
    //! Returns the number of virtual crystal rings per block
    inline size_t GetVirtualRingPerBlockNb() const
      { return m_virtualRingPerBlockNb;}
    //! Set the number of rings
    inline void SetVirtualRingPerBlockNb(size_t aNb)
      { m_virtualRingPerBlockNb = aNb;}

    // C. Comtat, February 2011: Required to simulate Biograph output sinograms with virtual crystals
    //! Returns the number of virtual transaxial crystal per block
    inline size_t GetVirtualCrystalPerBlockNb() const
      { return m_virtualCrystalPerBlockNb;}
    //! Set the number of rings
    inline void SetVirtualCrystalPerBlockNb(size_t aNb)
      { m_virtualCrystalPerBlockNb = aNb;}


     //! Returns the number of radial sinogram bins
     inline size_t GetRadialElemNb() const
       { return  m_radialElemNb;}
     //! Set the number of radial sinogram bin
     inline void SetRadialElemNb(size_t aNb)
       { m_radialElemNb = aNb;}

     //! Returns the number of 2D sinograms
     inline size_t GetSinogramNb() const
       { return  m_sinogramNb;}
     //! Set the number of radial sinogram bin
     inline void SetSinogramNb(size_t aNb)
       { m_sinogramNb = aNb;}

    //! Returns the number of crystals per crystal ring
    inline size_t GetCrystalNb() const
      { return m_crystalNb;}
    //! Set the number of crystals per crystal ring
    inline void SetCrystalNb(size_t aNb)
      { m_crystalNb = aNb;}

     //! Returns the data pointer
    inline SinogramDataType** GetData() const
      { return m_data;}

    //! Returns the randoms pointer
    inline SinogramDataType* GetRandoms() const
      { return m_randomsNb;}

    //! Set the verbose level
    virtual void SetVerboseLevel(G4int val)
      { nVerboseLevel = val; };

    //! Returns a 2D sinogram from the set
    inline SinogramDataType* GetSinogram(size_t sinoID) const
      { return m_data[sinoID];}

    //! Returns the number of pixels per 2D sinogram
    inline G4int PixelsPerSinogram() const
      { return m_radialElemNb * m_crystalNb / 2;}

     //! Returns the number of bytes per 2D sinogram
    inline G4int BytesPerSinogram() const
      { return PixelsPerSinogram() * BytesPerPixel() ;}

     //! Returns the nb of bytes per pixel
    inline size_t BytesPerPixel() const
      { return sizeof(SinogramDataType);}

     //! Returns the current frame ID
    inline size_t GetCurrentFrameID() const
      { return m_currentFrameID;}

     //! Returns the current gate ID
    inline size_t GetCurrentGateID() const
      { return m_currentGateID;}

     //! Returns the current data ID
    inline size_t GetCurrentDataID() const
      { return m_currentDataID;}

     //! Returns the current bed ID
    inline size_t GetCurrentBedID() const
      { return m_currentBedID;}


  //@}

    /*! \brief Writes a 2D sinogram onto an output stream

      	\param dest:    	  the destination stream
      	\param sinoID:    	  the 2D sinogram to stream-out
    */
    void StreamOut(std::ofstream& dest, size_t sinoID, size_t seekID);

    //! \name Data fields
    //@{

    size_t                m_ringNb;                             //!< Nb of crystal rings
    size_t		  m_crystalNb;                          //!< Nb of crystals per crystal ring
    SinogramDataType    **m_data;                               //!< Array of 2D sinograms
    G4int                 m_currentFrameID;                     //!< ID of the current frame (dynamique acquisitions)
    G4int                 m_currentGateID;                      //!< ID of the current gate (synchronized acquisitions)
    G4int 	          m_currentDataID;                      //!< ID of the current coincidence type (prompts or trues, delayed, LowEnergy, ...)
    G4int                 m_currentBedID;                       //!< ID of the current bed position (multi-bed acquisitions)
    G4int     	      	  nVerboseLevel;
    SinogramDataType     *m_randomsNb;                          //!< Total number of randoms per 2D sinogram
    size_t		  m_radialElemNb;			//!< Nb of radial sinogram bins
    size_t                m_sinogramNb;                         //!< Nb of 2D sinograms

    // C. Comtat, February 2011: Required to simulate Biograph output sinograms with virtual crystals
    size_t                m_virtualRingPerBlockNb;              //!< Nb of virtual crystal rings per block, used for sinogram output bin identifacation
    size_t                m_virtualCrystalPerBlockNb;           //!< Nb of virtual crystals in transaxial direction per block, used for sinogram output bin identifacation

    // ProjectionDataType   *m_dataMax;       	      	      	//!< Max count for each projection

    //@}

};




// Public constructor
inline GateSinogram::GateSinogram()
  : m_ringNb(0)
  , m_crystalNb(0)
  , m_data(0)
  , m_currentFrameID(-1)
  , m_currentGateID(-1)
  , m_currentDataID(-1)
  , m_currentBedID(-1)
  , nVerboseLevel(0)
  , m_randomsNb(0)
  , m_radialElemNb(0)
  , m_sinogramNb(0)
  , m_virtualRingPerBlockNb(0)
  , m_virtualCrystalPerBlockNb(0)
{
}


#endif
